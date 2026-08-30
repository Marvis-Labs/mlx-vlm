from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def finalize(
    job: Mapping[str, Any], result: Mapping[str, Any] | None
) -> dict[str, Any]:
    if result is None:
        return {
            "component": str(job.get("component", "runner")),
            "model": job.get("model"),
            "job_id": str(job.get("id", "")),
            "outcome": "infrastructure_failure",
            "findings": {"error": "runner produced no result"},
        }

    output = dict(result)
    findings = output.get("findings")
    if isinstance(findings, Mapping):
        phases = findings.get("phases")
        if isinstance(phases, Mapping):
            output["phases"] = dict(phases)
        metrics = findings.get("metrics")
        if isinstance(metrics, Mapping):
            output["metrics"] = dict(metrics)
        verdict = findings.get("verdict")
        if output.get("decision") == "accepted" and verdict in {
            "improved",
            "regressed",
            "test_failure",
        }:
            output["outcome"] = verdict
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    job = json.loads(args.job.read_text())
    raw_result = json.loads(args.result.read_text()) if args.result.is_file() else None
    result = finalize(job, raw_result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
