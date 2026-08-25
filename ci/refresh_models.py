"""Verify model metadata against the Hub.

Which model a cell downloads is declared, never discovered: searching at route
time would let the same pull request pick a different checkpoint on a
different day. This keeps the declaration honest instead.

    python -m ci.refresh_models            fill sha, size, gated
    python -m ci.refresh_models --check    fail if anything drifted
"""

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

import yaml

MODELS = Path(__file__).resolve().parent / "models.yaml"


def hub(repo: str) -> dict:
    d = json.load(
        urllib.request.urlopen(
            f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60
        )
    )
    gb = (
        sum(
            s.get("size") or 0
            for s in d["siblings"]
            if s["rfilename"].endswith((".safetensors", ".npz", ".bin"))
        )
        / 2**30
    )
    return {"sha": d["sha"], "weights_gb": round(gb, 2), "gated": bool(d.get("gated"))}


def main() -> int:
    check = "--check" in sys.argv
    doc = yaml.safe_load(MODELS.read_text())
    problems = []
    for arch, entry in doc.items():
        for v in entry["variants"]:
            try:
                info = hub(v["repo"])
            except Exception as exc:
                problems.append(f"{arch}/{v['repo']}: {type(exc).__name__}")
                continue
            # A moved revision means different weights, so any result measured
            # against the old one has to be re-baselined rather than compared.
            if v.get("sha") and v["sha"] != info["sha"]:
                problems.append(f"{arch}/{v['repo']}: revision moved, re-baseline")
            v.update(info, verified=str(date.today()))

    for p in problems:
        print(p, file=sys.stderr)
    n = sum(len(e["variants"]) for e in doc.values())
    if check:
        print(f"ok: {n} variants" if not problems else "", file=sys.stderr)
        return 1 if problems else 0
    MODELS.write_text(yaml.safe_dump(doc, sort_keys=True, default_flow_style=False))
    print(f"updated {n} variants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
