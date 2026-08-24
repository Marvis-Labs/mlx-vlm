"""Regenerate the recorded capability matrix.

Run this when a change is meant to alter what an architecture can be run with,
and commit the result alongside it:

    python -m mlx_vlm.tests.generate_capabilities

Only architectures resolvable on this machine are rewritten. Rows for
architectures whose config is not cached here are left as they are, so
regenerating on one machine does not discard another's coverage.
"""

import json
import os
from dataclasses import asdict

import mlx.core as mx

from mlx_vlm.tests.capabilities import capabilities
from mlx_vlm.tests.models_registry import _cached_configs

RECORD = os.path.join(os.path.dirname(__file__), "capabilities.json")


def architectures():
    models = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    present = {
        name
        for name in os.listdir(models)
        if os.path.isdir(os.path.join(models, name)) and name != "__pycache__"
    }
    return sorted(present & set(_cached_configs()))


def main() -> int:
    recorded = {}
    if os.path.exists(RECORD):
        recorded = json.load(open(RECORD))

    probed, failed = 0, []
    for arch in architectures():
        try:
            caps = capabilities(arch)
        except Exception as error:  # unresolvable here; leave any existing row alone
            failed.append((arch, type(error).__name__))
            continue
        row = asdict(caps)
        row.pop("arch")
        row["cache_kinds"] = sorted(set(row["cache_kinds"]))
        recorded[arch] = row
        probed += 1
        mx.clear_cache()

    with open(RECORD, "w") as handle:
        json.dump(dict(sorted(recorded.items())), handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"recorded {probed} architectures, {len(recorded)} rows total -> {RECORD}")
    for arch, error in failed:
        print(f"  skipped {arch}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
