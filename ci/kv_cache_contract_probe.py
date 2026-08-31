from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from ci.kv_cache_contract import ContractRunner
from ci.kv_cache_profiles import dense_contract_cases


def run(profiles: Sequence[str] = ("dense",)) -> dict[str, Any]:
    import mlx_vlm.models.cache as cache_module

    unsupported = sorted(set(profiles) - {"dense"})
    if unsupported:
        raise ValueError(f"unsupported cache profiles: {','.join(unsupported)}")
    cases = dense_contract_cases() if "dense" in profiles else ()
    results = [ContractRunner().run(case).to_dict() for case in cases]
    passed = bool(results) and all(result["verdict"] == "passed" for result in results)
    return {
        "component": "kv_cache_change",
        "profiles": list(profiles),
        "implementation_path": str(Path(cache_module.__file__).resolve()),
        "verdict": "passed" if passed else "test_failure",
        "checks": sum(int(result["checks"]) for result in results),
        "cases": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(tuple(args.profiles or ("dense",)))
    output = args.output or Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
