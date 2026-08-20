#!/usr/bin/env bash
# Give the active docker-backed trainer a bounded chance to flush its current
# recovery boundary before systemd's 55-second stop deadline expires.  SIGTERM
# is intentional: Docker forwards it to the trainer instead of discarding its
# local 500-step checkpoint/recovery cursor.
set -euo pipefail

if [[ "${1:-}" != "stop" || $# -ne 1 ]]; then
  echo "usage: lehome-training-control.sh stop" >&2
  exit 2
fi

PID_FILE="${LEHOME_TRAINING_PID_FILE:-/run/lehome-training.pid}"
if [[ -L "${PID_FILE}" || ! -f "${PID_FILE}" ]]; then
  exit 0
fi
pid="$(tr -d '[:space:]' < "${PID_FILE}")"
if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "unsafe training pid file" >&2
  exit 1
fi
if ! kill -0 "${pid}" 2>/dev/null; then
  exit 0
fi
kill -TERM "${pid}"
for _ in $(seq 1 45); do
  kill -0 "${pid}" 2>/dev/null || exit 0
  sleep 1
done
echo "trainer did not stop within the bounded preemption window" >&2
exit 1
