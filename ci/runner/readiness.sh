#!/bin/bash
# Refuse work rather than produce a measurement that is quietly wrong.
# Exit 0 = ready. Exit 1 = not ready, reason on stdout.
set -uo pipefail
NEED_GB="${NEED_GB:-8}"
MAX_LOAD="${MAX_LOAD:-2.0}"
fail() { echo "NOT READY: $*"; exit 1; }

# Nothing else may be running: benchmarks are single-tenant by definition.
STRAY=$(pgrep -fl 'python|mlx' | grep -v readiness | grep -vc Runner.Listener || true)
[ "$STRAY" -gt 0 ] && fail "$STRAY stray python/mlx process(es)"

LOAD=$(sysctl -n vm.loadavg | awk '{print $2}')
awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}' && fail "load $LOAD exceeds $MAX_LOAD"

# Free + inactive pages are both reclaimable for unified memory.
PAGE=$(pagesize)
FREE_GB=$(vm_stat | awk -v p="$PAGE" '/Pages free/{f=$3} /Pages inactive/{i=$3} END{printf "%.1f",(f+i)*p/1073741824}')
awk -v f="$FREE_GB" -v n="$NEED_GB" 'BEGIN{exit !(f<n)}' && fail "free memory ${FREE_GB}GB below ${NEED_GB}GB"

SWAP=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6+0}')
awk -v s="$SWAP" 'BEGIN{exit !(s>512)}' && fail "swap in use: ${SWAP}MB"

DISK_GB=$(df -g / | awk 'NR==2{print $4}')
[ "$DISK_GB" -lt 50 ] && fail "disk free ${DISK_GB}GB below 50GB"

# CPU_Speed_Limit < 100 means the SoC is thermally throttled right now.
THERM=$(pmset -g therm 2>/dev/null | awk -F'= ' '/CPU_Speed_Limit/{print $2+0}')
[ -n "${THERM:-}" ] && [ "$THERM" -lt 100 ] && fail "thermally throttled (CPU_Speed_Limit=$THERM)"

if pmset -g batt 2>/dev/null | grep -q "Battery Power"; then fail "on battery, not mains"; fi

echo "READY load=$LOAD free=${FREE_GB}GB disk=${DISK_GB}GB therm=${THERM:-na}"
