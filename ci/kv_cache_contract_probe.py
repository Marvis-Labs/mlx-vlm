from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from ci.kv_cache_contract import ContractRunner

DEFAULT_CONTRACT = "ci.kv_cache_profiles.dense:dense_contract_cases"


def run(contracts: Sequence[str] = (DEFAULT_CONTRACT,)) -> dict[str, Any]:
    import mlx_vlm.models.cache as cache_module

    cases = tuple(
        case for entry_point in contracts for case in _load_contract(entry_point)()
    )
    results = [ContractRunner().run(case).to_dict() for case in cases]
    passed = bool(results) and all(result["verdict"] == "passed" for result in results)
    return {
        "component": "kv_cache_change",
        "profiles": sorted({result["profile"] for result in results}),
        "implementation_path": str(Path(cache_module.__file__).resolve()),
        "verdict": "passed" if passed else "test_failure",
        "checks": sum(int(result["checks"]) for result in results),
        "cases": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", action="append", dest="contracts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(tuple(args.contracts or (DEFAULT_CONTRACT,)))
    output = args.output or Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["verdict"] == "passed" else 2


def _load_contract(entry_point: str):
    module_name, separator, function_name = entry_point.partition(":")
    if (
        separator != ":"
        or not module_name.startswith("ci.kv_cache_profiles.")
        or not function_name.endswith("_contract_cases")
    ):
        raise ValueError(f"invalid KV cache contract entry point: {entry_point}")
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"KV cache contract is not callable: {entry_point}")
    return function


if __name__ == "__main__":
    raise SystemExit(main())
