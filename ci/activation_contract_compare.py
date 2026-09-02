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
        raise RuntimeError("activation contract probe output must be an object")
    return value


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "verdict": str(result.get("verdict", "test_failure")),
        "checks": int(summary.get("checks", 0)),
        "failures": list(summary.get("failures", [])),
        "output_hash": result.get("output_hash"),
        "error": result.get("error"),
    }


def compare(base: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]:
    base_summary = _summary(base)
    head_summary = _summary(head)
    base_passed = base_summary["verdict"] == "passed"
    head_passed = head_summary["verdict"] == "passed"
    return {
        "verdict": "passed" if head_passed else "test_failure",
        "correctness": {
            "match": head_passed,
            "base_contract": "passed" if base_passed else "failed",
            "head_contract": "passed" if head_passed else "failed",
            "behavior_changed": base_summary.get("output_hash")
            != head_summary.get("output_hash"),
        },
        "base": base_summary,
        "head": head_summary,
    }


def _failed_probe(error: Exception) -> dict[str, Any]:
    return {
        "verdict": "unavailable",
        "summary": {"checks": 0, "failures": []},
        "error": f"{type(error).__name__}: {error}",
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
    except Exception as error:
        base = _failed_probe(error)
    try:
        head = run_probe(
            args.head, args.probe, args.job, findings.with_suffix(".head.json")
        )
    except Exception as error:
        head = _failed_probe(error)
    result = compare(base, head)
    findings.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 2 if result["verdict"] == "test_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
