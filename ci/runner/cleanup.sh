#!/bin/bash
# Always runs, including after a failure. Keeps the weight cache: it is
# revision-pinned and re-downloading it is pure cost.
set -uo pipefail
WORK="${CI_WORK:-$HOME/ci-work}"
REPO="${CI_REPO:-$PWD}"
here="$(cd "$(dirname "$0")" && pwd)"

"$here/quiesce.sh" release || true
for tree in "$WORK"/rev-*; do
  [ -d "$tree" ] || continue
  git -C "$REPO" worktree remove --force "$tree" 2>/dev/null || rm -rf "$tree"
done
git -C "$REPO" worktree prune 2>/dev/null || true
printf '\033[1;34m[cleanup]\033[0m released\n'
