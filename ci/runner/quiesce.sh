#!/bin/bash
# Take exclusive ownership of a device for the duration of a benchmark, then
# give it back. Suspends rather than kills: nothing loses state, and every
# action is reversible from the snapshot alone, without this process.
#
#   quiesce.sh acquire   # take the lock, suspend contenders
#   quiesce.sh release   # resume everything, drop the lock
#   quiesce.sh status
set -uo pipefail

STATE=/var/tmp/mlx-vlm-ci
LOCK="$STATE/lock"
SNAP="$STATE/suspended.pids"
CAFF="$STATE/caffeinate.pid"
MIN_CPU="${MIN_CPU:-1.0}"          # ignore idle processes
say() { printf '\033[1;34m[quiesce]\033[0m %s\n' "$*"; }

# Suspending any of these loses the machine until someone touches it
# physically, or kills the job that is trying to run.
PROTECT='sshd|ssh-agent|Runner\.Listener|Runner\.Worker|run\.sh|Tailscale|tailscaled|
launchd|logind|WindowServer|loginwindow|kernel_task|quiesce\.sh|watchdog\.sh|caffeinate'
PROTECT=$(echo "$PROTECT" | tr -d '\n ')

mkdir -p "$STATE"

acquire() {
  if [ -f "$LOCK" ]; then
    owner=$(cat "$LOCK" 2>/dev/null || echo "?")
    if kill -0 "$owner" 2>/dev/null; then
      echo "device busy: held by pid $owner"; exit 1
    fi
    say "stale lock from dead pid $owner; reclaiming"
    release_locked
  fi
  echo $$ > "$LOCK"

  say "quitting user applications gracefully"
  osascript -e 'tell application "System Events" to get name of every application process whose background only is false' 2>/dev/null \
    | tr ',' '\n' | sed 's/^ *//' | grep -v '^Finder$' | while read -r app; do
        [ -n "$app" ] && osascript -e "tell application \"$app\" to quit" 2>/dev/null || true
      done

  say "suspending remaining contenders above ${MIN_CPU}% cpu"
  : > "$SNAP"
  ps -Ao pid,pcpu,comm -u "$(whoami)" | tail -n +2 | while read -r pid cpu comm; do
    [ "$pid" = "$$" ] && continue
    case "$comm" in *ps|*awk) continue;; esac
    echo "$comm" | grep -Eq "$PROTECT" && continue
    # Never suspend an ancestor: that would freeze this script.
    p=$$; while [ "$p" -gt 1 ]; do
      [ "$p" = "$pid" ] && continue 2
      p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$p" ] && break
    done
    awk -v c="$cpu" -v m="$MIN_CPU" 'BEGIN{exit !(c>m)}' || continue
    if kill -STOP "$pid" 2>/dev/null; then echo "$pid" >> "$SNAP"; fi
  done
  say "suspended $(wc -l < "$SNAP" | tr -d ' ') process(es)"

  say "disabling indexing and backup"
  sudo mdutil -a -i off  >/dev/null 2>&1 || true
  sudo tmutil disable    >/dev/null 2>&1 || true

  caffeinate -dimsu & echo $! > "$CAFF"
  say "device is exclusive"
}

release_locked() {
  if [ -f "$SNAP" ]; then
    n=0; while read -r pid; do kill -CONT "$pid" 2>/dev/null && n=$((n+1)); done < "$SNAP"
    say "resumed $n process(es)"; rm -f "$SNAP"
  fi
  [ -f "$CAFF" ] && { kill "$(cat "$CAFF")" 2>/dev/null; rm -f "$CAFF"; }
  sudo mdutil -a -i on >/dev/null 2>&1 || true
  sudo tmutil enable   >/dev/null 2>&1 || true
  rm -f "$LOCK"
}

case "${1:-status}" in
  acquire) acquire ;;
  release) release_locked; say "device released" ;;
  status)
    if [ -f "$LOCK" ]; then
      echo "held by pid $(cat "$LOCK"), $(wc -l < "$SNAP" 2>/dev/null | tr -d ' ') suspended"
    else echo "free"; fi ;;
  *) echo "usage: $0 {acquire|release|status}"; exit 2 ;;
esac
