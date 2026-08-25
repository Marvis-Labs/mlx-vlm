#!/bin/bash
# Thin wrapper: everything that reasons lives in Python.
#   measure.sh <cell.json> <base-tree> <head-tree>
set -euo pipefail
CELL="$1"; BASE_TREE="$2"; HEAD_TREE="$3"
exec python3 -m ci.cell \
  --cell "$CELL" --base-tree "$BASE_TREE" --head-tree "$HEAD_TREE" \
  --repeats "${CI_REPEATS:-3}" --warmup "${CI_WARMUP:-1}" \
  --out "${CI_RESULT:-result.json}"
