#!/bin/bash
# Provision a bare macOS device as an mlx-vlm-ci benchmark runner.
# Idempotent: safe to re-run. Requires sudo for the system policy section.
#
#   REPO=Marvis-Labs/mlx-vlm-ci RUNNER_TOKEN=... ./bootstrap.sh
set -euo pipefail

REPO="${REPO:-Marvis-Labs/mlx-vlm-ci}"
RUNNER_VERSION="${RUNNER_VERSION:-2.336.0}"
RUNNER_DIR="$HOME/actions-runner"
WORK_DIR="$HOME/ci-work"
say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- labels
CHIP=$(sysctl -n machdep.cpu.brand_string | tr 'A-Z ' 'a-z-')   # apple-m4-pro
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
LABELS="self-hosted,macos,arm64,${CHIP},mem-${MEM_GB}"
[ "$MEM_GB" -ge 96 ] && LABELS="${LABELS},parity"   # room for two model copies
say "labels: $LABELS"

# ---------------------------------------------------------------- toolchain
if ! xcode-select -p >/dev/null 2>&1; then
  say "installing Command Line Tools (headless)"
  touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
  LABEL=$(softwareupdate -l 2>/dev/null | grep -o 'Command Line Tools for Xcode-[0-9.]*' | tail -1)
  [ -n "$LABEL" ] && sudo softwareupdate -i "$LABEL" --verbose
  rm -f /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
fi

if ! command -v uv >/dev/null 2>&1; then
  say "installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------- system policy
say "applying power, indexing and memory policy"
sudo pmset -a sleep 0 disksleep 0 displaysleep 10 powernap 0 womp 1
sudo pmset -a lowpowermode 0 2>/dev/null || true

mkdir -p "$WORK_DIR" "$HOME/.cache/huggingface"
sudo mdutil -i off "$WORK_DIR" >/dev/null 2>&1 || true
sudo mdutil -i off "$HOME/.cache" >/dev/null 2>&1 || true
tmutil addexclusion "$WORK_DIR" 2>/dev/null || true
tmutil addexclusion "$HOME/.cache" 2>/dev/null || true

# Wired-memory limit: non-persistent sysctl, so reapply at every boot.
WIRED_MB=$(( MEM_GB * 1024 * 90 / 100 ))
sudo /usr/libexec/PlistBuddy -c "Clear dict" \
  -c "Add :Label string com.marvis.ci.wiredlimit" \
  -c "Add :RunAtLoad bool true" \
  -c "Add :ProgramArguments array" \
  -c "Add :ProgramArguments:0 string /usr/sbin/sysctl" \
  -c "Add :ProgramArguments:1 string -w" \
  -c "Add :ProgramArguments:2 string iogpu.wired_limit_mb=${WIRED_MB}" \
  /Library/LaunchDaemons/com.marvis.ci.wiredlimit.plist >/dev/null
sudo launchctl bootout system/com.marvis.ci.wiredlimit 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.marvis.ci.wiredlimit.plist
say "wired limit set to ${WIRED_MB} MB of ${MEM_GB} GB"

# ---------------------------------------------------------------- runner
if [ ! -x "$RUNNER_DIR/run.sh" ]; then
  say "installing Actions runner $RUNNER_VERSION"
  mkdir -p "$RUNNER_DIR" && cd "$RUNNER_DIR"
  curl -fsSLo runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
  tar xzf runner.tar.gz && rm runner.tar.gz
fi

cd "$RUNNER_DIR"
if [ ! -f .runner ]; then
  : "${RUNNER_TOKEN:?set RUNNER_TOKEN (gh api -X POST repos/$REPO/actions/runners/registration-token)}"
  say "registering with $REPO"
  ./config.sh --unattended --replace \
    --url "https://github.com/${REPO}" --token "$RUNNER_TOKEN" \
    --name "$(hostname -s)" --labels "$LABELS" --work "$WORK_DIR"
fi

say "installing runner as a launchd service"
sudo ./svc.sh install "$(whoami)" >/dev/null
sudo ./svc.sh start

say "done — $(hostname -s) is registered and serving"
