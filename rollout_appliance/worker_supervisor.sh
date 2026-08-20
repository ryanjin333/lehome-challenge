#!/usr/bin/env bash
# Small testable supervision primitives for the persistent Isaac containers.

lehome_supervise_worker() {
  local index="$1"
  local max_restarts="$2"
  local launch_function="$3"
  local restarts=0
  local retry_delay_seconds="${LEHOME_WORKER_RESTART_DELAY_SECONDS:-2}"

  while true; do
    # A zero exit means lease_next() found no eligible work.  It is a clean
    # drain, never a restart condition.
    if "${launch_function}" "${index}"; then
      return 0
    fi
    if (( restarts >= max_restarts )); then
      echo "worker ${index} exceeded restart limit (${max_restarts}); failing campaign closed" >&2
      return 70
    fi
    restarts=$((restarts + 1))
    echo "worker ${index} exited unexpectedly; restarting ${restarts}/${max_restarts}" >&2
    sleep "${retry_delay_seconds}"
  done
}

lehome_worker_exit_is_clean() {
  local database="$1"
  local worker_id="$2"
  python3 - "${database}" "${worker_id}" <<'PY'
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1])
try:
    connection.execute("PRAGMA query_only = ON")
    active = connection.execute(
        "SELECT COUNT(*) FROM events AS event "
        "WHERE event.worker_id = ? "
        "AND event.event_type IN ('leased', 'heartbeat') "
        "AND event.event_id = ("
        "SELECT MAX(later.event_id) FROM events AS later "
        "WHERE later.attempt_id = event.attempt_id"
        ")",
        (sys.argv[2],),
    ).fetchone()[0]
finally:
    connection.close()
raise SystemExit(0 if active == 0 else 1)
PY
}

lehome_cleanup_policy() {
  local policy_pid="$1"
  if [[ "${policy_pid}" =~ ^[0-9]+$ ]]; then
    kill "${policy_pid}" 2>/dev/null || true
  fi
  docker rm -f lehome-12k-policy >/dev/null 2>&1 || true
}
