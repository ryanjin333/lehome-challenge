#!/usr/bin/env bash
# LeHome GR00T rollout entrypoint.
#
# Subcommands:
#   rollout   hydrate BC checkpoint + assets, serve policy, run headless eval
#   server    hydrate checkpoint and run only the GR00T policy server
#   <other>   exec the arguments as a command (interactive/debug sessions)
set -euo pipefail

LEHOME_ROOT="${LEHOME_ROOT:-/opt/lehome-challenge}"
GROOT_PYTHON="${GROOT_PYTHON:-/opt/gr00t-runtime/bin/python}"
POLICY_SERVER_SCRIPT="${POLICY_SERVER_SCRIPT:-/opt/lehome-rollout/groot_policy_server.py}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8080}"
GROOT_POLICY_DEVICE="${GROOT_POLICY_DEVICE:-cuda:0}"

# Initial BC run: ryanjin333/lehome-groot-n17-models @ policies/step-12000
# (repo head 30ac1a84da67b099e115ad147bcd61e9d60046d3 on 2026-08-04).
POLICY_REPO="${POLICY_REPO:-ryanjin333/lehome-groot-n17-models}"
POLICY_REVISION="${POLICY_REVISION:-30ac1a84da67b099e115ad147bcd61e9d60046d3}"
POLICY_INCLUDE="${POLICY_INCLUDE:-policies/step-12000/*}"
POLICY_MODEL_SUBDIR="${POLICY_MODEL_SUBDIR:-policies/step-12000}"
ASSET_REPO="${ASSET_REPO:-lehome/asset_challenge}"

# GR00T N1.7 backbone. The checkpoint config.json hard-references the exact
# local path below (config.model_name), so hydrate Cosmos-Reason2-2B there.
BACKBONE_REPO="${BACKBONE_REPO:-nvidia/Cosmos-Reason2-2B}"
BACKBONE_LOCAL_PATH="${BACKBONE_LOCAL_PATH:-/cache/models/nvidia/Cosmos-Reason2-2B}"

ROLLOUT_ROOT="${ROLLOUT_ROOT:-/workspace/rollout}"
GARMENT_TYPES="${GARMENT_TYPES:-top_long top_short pant_long pant_short}"
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
READINESS_TIMEOUT_SECONDS="${READINESS_TIMEOUT_SECONDS:-1800}"

LOG_DIR="${ROLLOUT_ROOT}/logs"
POLICY_SERVER_PID=""

log() { printf '[lehome-rollout] %s\n' "$*"; }
die() { printf '[lehome-rollout] ERROR: %s\n' "$*" >&2; exit 1; }

hf_cli() {
  if [[ -x "${LEHOME_ROOT}/.venv/bin/hf" ]]; then
    "${LEHOME_ROOT}/.venv/bin/hf" "$@"
  else
    hf "$@"
  fi
}

lehome_python() {
  if [[ -x "${LEHOME_ROOT}/.venv/bin/python" ]]; then
    "${LEHOME_ROOT}/.venv/bin/python" "$@"
  else
    python3 "$@"
  fi
}

require_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is required (private policy repository ${POLICY_REPO})."
}

cleanup() {
  if [[ -n "${POLICY_SERVER_PID}" ]] && kill -0 "${POLICY_SERVER_PID}" 2>/dev/null; then
    log "stopping policy server (pid ${POLICY_SERVER_PID})"
    kill "${POLICY_SERVER_PID}" 2>/dev/null || true
    wait "${POLICY_SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

hydrate_policy_checkpoint() {
  require_hf_token
  local repo_root="${ROLLOUT_ROOT}/policy-repo"
  mkdir -p "${repo_root}"
  log "hydrating policy checkpoint: ${POLICY_REPO}@${POLICY_REVISION} (include: ${POLICY_INCLUDE})"
  HF_TOKEN="${HF_TOKEN}" hf_cli download "${POLICY_REPO}" \
    --revision "${POLICY_REVISION}" \
    --include "${POLICY_INCLUDE}" \
    --local-dir "${repo_root}"

  local model_path
  if [[ -n "${POLICY_MODEL_SUBDIR}" ]]; then
    model_path="${repo_root}/${POLICY_MODEL_SUBDIR}"
    [[ -d "${model_path}" ]] || die "POLICY_MODEL_SUBDIR not found: ${model_path}"
  else
    # Auto-detect a GR00T checkpoint: prefer dirs with model.safetensors +
    # experiment_cfg, fall back to model.safetensors alone, newest first.
    model_path="$(
      {
        find "${repo_root}" -name model.safetensors 2>/dev/null
        find "${repo_root}" -name model.safetensors.index.json 2>/dev/null
        find "${repo_root}" -name 'model-00001-of-*.safetensors' 2>/dev/null
      } | while read -r file; do
            dir="$(dirname "${file}")"
            if [[ -d "${dir}/experiment_cfg" ]]; then printf '1\t%s\n' "${dir}"; else printf '2\t%s\n' "${dir}"; fi
          done \
        | sort -u | head -n1 | cut -f2-
    )"
    [[ -n "${model_path}" ]] || {
      find "${repo_root}" -maxdepth 3 -type d >&2 || true
      die "no GR00T checkpoint (model.safetensors[.index.json] or shards) found under ${repo_root}; set POLICY_MODEL_SUBDIR"
    }
  fi
  log "policy checkpoint resolved to ${model_path}"
  POLICY_MODEL_PATH="${model_path}"
}

hydrate_backbone() {
  if [[ -f "${BACKBONE_LOCAL_PATH}/config.json" ]]; then
    log "backbone already present at ${BACKBONE_LOCAL_PATH}"
    return 0
  fi
  log "hydrating GR00T backbone ${BACKBONE_REPO} -> ${BACKBONE_LOCAL_PATH}"
  mkdir -p "$(dirname "${BACKBONE_LOCAL_PATH}")"
  hf_cli download "${BACKBONE_REPO}" --local-dir "${BACKBONE_LOCAL_PATH}"
  [[ -f "${BACKBONE_LOCAL_PATH}/config.json" ]] || die "backbone hydration failed: ${BACKBONE_LOCAL_PATH}/config.json missing"
}

ensure_assets() {
  if [[ -d "${LEHOME_ROOT}/Assets/objects" ]]; then
    log "simulation assets already present"
    return 0
  fi
  log "downloading simulation assets from ${ASSET_REPO}"
  hf_cli download "${ASSET_REPO}" --repo-type dataset --local-dir "${LEHOME_ROOT}/Assets"
}

start_policy_server() {
  mkdir -p "${LOG_DIR}"
  log "starting GR00T policy server on 127.0.0.1:${POLICY_SERVER_PORT} (${GROOT_POLICY_DEVICE})"
  PYTHONPATH="${GROOT_ROOT:-/opt/isaac-groot}" "${GROOT_PYTHON}" "${POLICY_SERVER_SCRIPT}" \
    --model-path "${POLICY_MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${POLICY_SERVER_PORT}" \
    --device "${GROOT_POLICY_DEVICE}" \
    >>"${LOG_DIR}/policy-server.log" 2>&1 &
  POLICY_SERVER_PID=$!
  log "policy server pid ${POLICY_SERVER_PID}"
}

await_policy_server() {
  log "waiting up to ${READINESS_TIMEOUT_SECONDS}s for policy server readiness"
  lehome_python - "${READINESS_TIMEOUT_SECONDS}" "${POLICY_SERVER_PORT}" "${POLICY_SERVER_PID}" <<'PY'
import sys
import time
import urllib.request

timeout_seconds = int(sys.argv[1])
port = int(sys.argv[2])
server_pid = int(sys.argv[3])
deadline = time.monotonic() + timeout_seconds
url = f"http://127.0.0.1:{port}/reset"
request_body = b"{}"

while time.monotonic() < deadline:
    try:
        with open(f"/proc/{server_pid}") as handle:
            handle.read()
    except OSError:
        raise SystemExit("policy server process exited before becoming ready")
    try:
        request = urllib.request.Request(
            url, data=request_body, method="POST", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                print(f"[lehome-rollout] policy server ready at {url}", flush=True)
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as error:
        print(f"[lehome-rollout] policy server not ready yet: {error}", flush=True)
    time.sleep(5)
raise SystemExit(f"policy server did not become ready within {timeout_seconds}s")
PY
}

run_eval() {
  mkdir -p "${LOG_DIR}" "${ROLLOUT_ROOT}/results"
  local results_file="${ROLLOUT_ROOT}/rollout-results.txt"
  {
    printf 'LeHome GR00T rollout results\n'
    printf 'generated_at: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'policy_repo: %s\n' "${POLICY_REPO}"
    printf 'policy_revision: %s\n' "${POLICY_REVISION}"
    printf 'policy_model_path: %s\n' "${POLICY_MODEL_PATH}"
    printf 'num_episodes: %s\n' "${NUM_EPISODES}"
    printf 'max_steps: %s\n' "${MAX_STEPS}"
    printf '\n'
  } >"${results_file}"

  local garment_type status=0
  for garment_type in ${GARMENT_TYPES}; do
    local eval_log="${LOG_DIR}/eval-${garment_type}.log"
    local video_args=()
    if [[ "${SAVE_VIDEO}" == "1" ]]; then
      video_args=(--save_video --video_dir "${ROLLOUT_ROOT}/videos/${garment_type}")
    fi
    log "evaluating garment_type=${garment_type} (num_episodes=${NUM_EPISODES})"
    if (
      cd "${LEHOME_ROOT}" && \
      lehome_python -m scripts.eval \
        --policy_type docker \
        --docker_url "http://127.0.0.1:${POLICY_SERVER_PORT}" \
        --garment_type "${garment_type}" \
        --num_episodes "${NUM_EPISODES}" \
        --max_steps "${MAX_STEPS}" \
        --enable_cameras \
        --device cpu \
        --headless \
        "${video_args[@]}"
    ) >"${eval_log}" 2>&1; then
      log "eval completed for ${garment_type}; log: ${eval_log}"
    else
      log "eval FAILED for ${garment_type}; log: ${eval_log}"
      status=1
    fi
    {
      printf '== garment_type: %s ==\n' "${garment_type}"
      grep -i -E "success|score|result|episode" "${eval_log}" | tail -n 40 || true
      printf '\n'
    } >>"${results_file}"
  done

  log "results written to ${results_file}"
  if [[ -n "${PUSH_REPO:-}" ]]; then
    require_hf_token
    log "pushing rollout artifacts to ${PUSH_REPO}"
    HF_TOKEN="${HF_TOKEN}" hf_cli upload "${PUSH_REPO}" \
      "${ROLLOUT_ROOT}/rollout-results.txt" "rollouts/$(date -u +%Y%m%dT%H%M%SZ)/rollout-results.txt"
    HF_TOKEN="${HF_TOKEN}" hf_cli upload "${PUSH_REPO}" \
      "${LOG_DIR}/." "rollouts/$(date -u +%Y%m%dT%H%M%SZ)/logs/" --repo-type dataset || true
  fi
  return "${status}"
}

command="${1:-rollout}"
case "${command}" in
  rollout)
    hydrate_policy_checkpoint
    hydrate_backbone
    ensure_assets
    start_policy_server
    await_policy_server
    run_eval
    ;;
  server)
    hydrate_policy_checkpoint
    hydrate_backbone
    start_policy_server
    log "policy server running; waiting for it to exit"
    wait "${POLICY_SERVER_PID}"
    ;;
  *)
    exec "$@"
    ;;
esac
