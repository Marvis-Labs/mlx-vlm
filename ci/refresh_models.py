"""Verify and refresh model metadata against the Hub.

Which model a cell downloads is declared, never discovered. Searching the Hub
at route time would let the same pull request pick different checkpoints on
different days, and two runs of the same cell would not be comparable. This
script keeps the declaration honest instead: it confirms every repository
exists, records its size, and pins the revision it was measured at.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import yaml

MODELS = Path(__file__).resolve().parent / "models.yaml"
WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin")


def hub_info(repo: str) -> dict:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as fh:
        d = json.load(fh)
    gb = (
        sum(
            s.get("size") or 0
            for s in d.get("siblings", [])
            if s["rfilename"].endswith(WEIGHT_SUFFIXES)
        )
        / 2**30
    )
    return {
        "sha": d["sha"],
        "weights_gb": round(gb, 2),
        "gated": bool(d.get("gated")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="verify only; fail if anything drifted"
    )
    args = ap.parse_args()

    doc = yaml.safe_load(MODELS.read_text())
    drift, missing = [], []

    for arch, entry in doc.items():
        for v in entry["variants"]:
            repo = v["repo"]
            try:
                info = hub_info(repo)
            except urllib.error.HTTPError as e:
                missing.append(f"{arch}/{repo}: HTTP {e.code}")
                continue
            except Exception as e:  # network, timeout
                missing.append(f"{arch}/{repo}: {type(e).__name__}")
                continue

            if v.get("sha") and v["sha"] != info["sha"]:
                drift.append(
                    f"{arch}/{repo}: revision moved "
                    f"{v['sha'][:8]} -> {info['sha'][:8]}"
                )
            if v.get("weights_gb") and abs(v["weights_gb"] - info["weights_gb"]) > 0.01:
                drift.append(
                    f"{arch}/{repo}: size {v['weights_gb']} -> "
                    f"{info['weights_gb']} GB"
                )
            v.update(info)
            v["verified"] = str(date.today())

    for line in missing:
        print(f"MISSING  {line}", file=sys.stderr)
    for line in drift:
        print(f"DRIFT    {line}", file=sys.stderr)

    if args.check:
        if missing or drift:
            print(
                "\nrun without --check to update, and re-baseline any "
                "affected results: a moved revision means different weights.",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {sum(len(e['variants']) for e in doc.values())} variants verified")
        return 0

    MODELS.write_text(yaml.safe_dump(doc, sort_keys=True, default_flow_style=False))
    print(f"updated {sum(len(e['variants']) for e in doc.values())} variants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
