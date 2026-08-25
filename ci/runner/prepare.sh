#!/bin/bash
# Two worktrees, one environment. mlx-vlm is pure Python, so revisions are a
# PYTHONPATH swap; sharing one pinned venv guarantees both halves see
# identical dependencies, which is the whole point of the comparison.
#
#   prepare.sh <base-sha> <head-sha>
set -euo pipefail
BASE="$1"; HEAD="$2"
WORK="${CI_WORK:-$HOME/ci-work}"
REPO="${CI_REPO:-$PWD}"
say() { printf '\033[1;34m[prepare]\033[0m %s\n' "$*"; }

mkdir -p "$WORK"
for rev in "$BASE" "$HEAD"; do
  tree="$WORK/rev-$rev"
  if [ ! -d "$tree" ]; then
    say "worktree for ${rev:0:8}"
    git -C "$REPO" worktree add -q --detach "$tree" "$rev"
  fi
done

# A dependency change invalidates the single-environment assumption: we would
# be measuring the dependency, not the diff. Refuse rather than mislead.
CHANGED=$(git -C "$REPO" diff --name-only "$BASE" "$HEAD" -- \
  pyproject.toml setup.py requirements.txt uv.lock 2>/dev/null || true)
if [ -n "$CHANGED" ]; then
  echo "REFUSING: dependency files differ between revisions:" >&2
  echo "$CHANGED" >&2
  echo "this cell needs two environments, not one" >&2
  exit 1
fi

echo "base_tree=$WORK/rev-$BASE"
echo "head_tree=$WORK/rev-$HEAD"
say "ready"
