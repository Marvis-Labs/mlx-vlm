#!/bin/bash
# What can this device actually hold?
#
# Reports the hardware, the benchmark budget derived from it, and which of
# the declared model variants fit. Run on a device before registering it, or
# any time you want to know why a cell was routed away from it.
#
#   capacity.sh            # human readable
#   capacity.sh --labels   # just the runner labels, for bootstrap
set -uo pipefail

FRACTION="${USABLE_FRACTION:-90}"        # percent reserved away from the OS
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$HERE/../models.yaml"

MEM_BYTES=$(sysctl -n hw.memsize)
MEM_GB=$(( MEM_BYTES / 1073741824 ))
BUDGET_GB=$(( MEM_GB * FRACTION / 100 ))
CHIP=$(sysctl -n machdep.cpu.brand_string)
GPU=$(system_profiler SPDisplaysDataType 2>/dev/null \
      | awk -F': ' '/Total Number of Cores/{print $2; exit}')
WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)

if [ "${1:-}" = "--labels" ]; then
  slug=$(echo "$CHIP" | tr 'A-Z ' 'a-z-')
  labels="self-hosted,macos,arm64,${slug},mem-${MEM_GB}"
  [ "$MEM_GB" -ge 96 ] && labels="${labels},parity"
  echo "$labels"; exit 0
fi

printf '\033[1m%s\033[0m\n' "$(hostname -s)"
printf '  chip            %s (%s GPU cores)\n' "$CHIP" "${GPU:-?}"
printf '  physical        %s GB\n' "$MEM_GB"
printf '  benchmark budget %s GB  (%s%% of physical)\n' "$BUDGET_GB" "$FRACTION"
if [ "$WIRED" != "0" ]; then
  printf '  wired limit     %s GB\n' "$(( WIRED / 1024 ))"
  if [ "$(( WIRED / 1024 ))" -lt "$BUDGET_GB" ]; then
    printf '  \033[1;33mwarning\033[0m       wired limit is below the budget; raise it or cells will swap\n'
  fi
else
  printf '  wired limit     unset  (bootstrap.sh installs the boot-time daemon)\n'
fi

# Free memory right now, which is what a cell actually gets.
PAGE=$(pagesize)
FREE_GB=$(vm_stat | awk -v p="$PAGE" '/Pages free/{f=$3} /Pages inactive/{i=$3} \
          END{printf "%.1f",(f+i)*p/1073741824}')
printf '  free right now  %s GB\n' "$FREE_GB"

[ -f "$MODELS" ] || { echo; echo "  (no models.yaml; cannot list what fits)"; exit 0; }

echo
python3 - "$MODELS" "$BUDGET_GB" "$FREE_GB" <<'PY'
import sys, yaml
models, budget, free = yaml.safe_load(open(sys.argv[1])), float(sys.argv[2]), float(sys.argv[3])

# Calibrated against measured peak memory rather than assumed. Overhead above
# the weights is roughly a fixed activation cost plus a cache term that grows
# with context, so the multiplier rises with prompt length:
#   184 tok -> 1.39x weights   730 -> 1.99x   2940 -> 1.90x   11832 -> 2.76x
# A cell is sized at the multiplier for its scenario, not a single constant.
MULT = {"short": 1.5, "typical": 2.0, "long": 2.8}

print(f"{'model':<58} {'weights':>8} {'typical':>8} {'long':>7}  fits")
print("-" * 96)
fit_typical = fit_long = total = 0
for arch, entry in sorted(models.items()):
    for v in entry["variants"]:
        w = v["weights_gb"]
        t, l = w * MULT["typical"], w * MULT["long"]
        ok_t, ok_l = t <= budget, l <= budget
        total += 1
        fit_typical += ok_t
        fit_long += ok_l
        mark = "yes" if ok_l else ("typical only" if ok_t else "NO")
        print(f"{v['repo'][:58]:<58} {w:>7.1f}G {t:>7.1f}G {l:>6.1f}G  {mark}")
print(f"\n  budget {budget:.0f} GB -> {fit_typical}/{total} variants at typical context, "
      f"{fit_long}/{total} at long context")
if free < budget:
    print(f"  note: only {free:.1f} GB free right now, so the effective ceiling is lower")
PY
