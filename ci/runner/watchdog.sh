#!/bin/bash
# Bash traps do not fire on SIGKILL, a panic, or a power loss. Without this,
# a harness that dies mid-run leaves the device with its processes suspended
# and indexing disabled, and only a physical visit fixes it.
#
# Installed as a launchd daemon, runs every 60s.
set -uo pipefail
STATE=/var/tmp/mlx-vlm-ci
LOCK="$STATE/lock"; SNAP="$STATE/suspended.pids"
MAX_AGE_MIN="${MAX_AGE_MIN:-120}"          # longest legitimate cell run

[ -f "$LOCK" ] || exit 0
owner=$(cat "$LOCK" 2>/dev/null || echo "")

reason=""
if [ -z "$owner" ] || ! kill -0 "$owner" 2>/dev/null; then
  reason="owner pid ${owner:-none} is gone"
elif [ -n "$(find "$LOCK" -mmin +"$MAX_AGE_MIN" 2>/dev/null)" ]; then
  reason="lock older than ${MAX_AGE_MIN} minutes"
fi
[ -z "$reason" ] && exit 0

logger -t mlx-vlm-ci "watchdog recovering device: $reason"
if [ -f "$SNAP" ]; then
  while read -r pid; do kill -CONT "$pid" 2>/dev/null; done < "$SNAP"
  rm -f "$SNAP"
fi
mdutil -a -i on >/dev/null 2>&1 || true
tmutil enable   >/dev/null 2>&1 || true
pkill -f 'caffeinate -dimsu' 2>/dev/null || true
rm -f "$LOCK"
logger -t mlx-vlm-ci "watchdog: device recovered"
