from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.probe_process import run_project_probe


def run_probe(project: Path, probe: Path, job: Path, output: Path) -> Mapping[str, Any]:
    run_project_probe(
        project,
        probe,
        [
            "--job",
            str(job),
            "--output",
            str(output),
        ],
    )
    value = json.loads(output.read_text())
    if not isinstance(value, Mapping):
        raise RuntimeError("MLP contract probe output must be an object")
    return value


def compare(base: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]:
    base_symbols = base.get("symbols", {})
    head_symbols = head.get("symbols", {})
    if not isinstance(base_symbols, Mapping) or not isinstance(head_symbols, Mapping):
        raise ValueError("MLP probe has no symbol results")
    checks: dict[str, Any] = {}
    for symbol in sorted(set(base_symbols) | set(head_symbols)):
        before = base_symbols.get(symbol, {})
        after = head_symbols.get(symbol, {})
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            checks[symbol] = {"match": False, "reason": "missing_symbol_result"}
            continue
        base_output = tuple(float(value) for value in before.get("output", []))
        head_output = tuple(float(value) for value in after.get("output", []))
        comparable = bool(base_output) and len(base_output) == len(head_output)
        max_abs_diff = (
            max(abs(left - right) for left, right in zip(base_output, head_output))
            if comparable
            else None
        )
        reference_diff = max(
            float(before.get("reference_max_abs_diff", float("inf"))),
            float(after.get("reference_max_abs_diff", float("inf"))),
        )
        match = bool(
            before.get("output_shape") == after.get("output_shape")
            and before.get("parameter_signature") == after.get("parameter_signature")
            and before.get("finite") is True
            and after.get("finite") is True
            and max_abs_diff is not None
            and max_abs_diff <= 1e-5
            and reference_diff <= 1e-5
        )
        checks[symbol] = {
            "match": match,
            "max_abs_diff": max_abs_diff,
            "reference_max_abs_diff": reference_diff,
            "base_output_hash": before.get("output_hash"),
            "head_output_hash": after.get("output_hash"),
        }
    match = bool(checks) and all(check["match"] for check in checks.values())
    return {
        "verdict": "passed" if match else "test_failure",
        "correctness": {"match": match, "symbols": checks},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args(argv)
    findings = Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    try:
        base = run_probe(
            args.base, args.probe, args.job, findings.with_suffix(".base.json")
        )
        head = run_probe(
            args.head, args.probe, args.job, findings.with_suffix(".head.json")
        )
        result = compare(base, head)
    except Exception as error:
        result = {
            "verdict": "test_failure",
            "error": f"{type(error).__name__}: {error}",
        }
    findings.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 2 if result["verdict"] == "test_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
