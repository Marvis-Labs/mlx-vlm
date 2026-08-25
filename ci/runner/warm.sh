#!/bin/bash
# Download the pinned weights once. Both halves of a comparison read the same
# warm cache; clearing between them would measure network throughput and
# report it as a regression.
#
#   warm.sh <cell.json>
set -euo pipefail
CELL="$1"
REPO=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['repo'])" "$CELL")
REV=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['revision'])" "$CELL")
printf '\033[1;34m[warm]\033[0m %s @ %s\n' "$REPO" "${REV:0:8}"

python3 - "$REPO" "$REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1], (sys.argv[2] or None)
path = snapshot_download(repo_id=repo, revision=rev,
                         allow_patterns=["*.json","*.safetensors","*.txt",
                                         "*.model","*.py","*.jinja"])
print(f"cached at {path}")
PY
