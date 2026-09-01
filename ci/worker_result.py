from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def finalize(
    job: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    infrastructure_error: str | None = None,
) -> dict[str, Any]:
    if infrastructure_error:
        output = dict(result or {})
        output.update(
            {
                "component": str(job.get("component", "runner")),
                "model": job.get("model"),
                "profile": job.get("profile"),
                "job_id": str(job.get("id", "")),
                "outcome": "infrastructure_failure",
                "findings": {"error": infrastructure_error},
            }
        )
        return output

    if result is None:
        return {
            "component": str(job.get("component", "runner")),
            "model": job.get("model"),
            "profile": job.get("profile"),
            "job_id": str(job.get("id", "")),
            "outcome": "infrastructure_failure",
            "findings": {"error": "runner produced no result"},
        }

    output = dict(result)
    output.setdefault("component", str(job.get("component", "runner")))
    output.setdefault("model", job.get("model"))
    output.setdefault("profile", job.get("profile"))
    output.setdefault("job_id", str(job.get("id", "")))
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
    parser.add_argument("--infrastructure-error", type=Path)
    args = parser.parse_args(argv)

    job = json.loads(args.job.read_text())
    raw_result = json.loads(args.result.read_text()) if args.result.is_file() else None
    infrastructure_error = None
    if args.infrastructure_error and args.infrastructure_error.is_file():
        error_lines = args.infrastructure_error.read_text().strip().splitlines()
        if error_lines:
            infrastructure_error = f"lease heartbeat failed: {error_lines[-1][:500]}"
    result = finalize(job, raw_result, infrastructure_error)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
