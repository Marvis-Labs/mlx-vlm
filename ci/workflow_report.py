from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.bot import BotOutput
from ci.scheduler import bot_result


def report(
    plan: Mapping[str, Any] | None,
    dispatch: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    run_url: str,
    attempt_id: str,
    head_sha: str,
    execution_status: str,
) -> dict[str, Any]:
    if result is not None:
        execution = dict(result)
    elif dispatch is not None and dispatch.get("outcome") == "no_eligible_runner":
        execution = bot_result(dispatch)
    else:
        job = dispatch.get("job", {}) if dispatch is not None else _job(plan)
        outcome = (
            "cancelled" if execution_status == "cancelled" else "infrastructure_failure"
        )
        selected_device = dispatch.get("next_device") if dispatch is not None else None
        device_name = (
            selected_device.get("name")
            if isinstance(selected_device, Mapping)
            else "selected runner"
        )
        execution = {
            "component": str(job.get("component", "runner")),
            "model": job.get("model"),
            "mode": str(job.get("mode", "default")),
            "job_id": str(job.get("id", "")),
            "outcome": outcome,
            "selected_device": selected_device,
            "findings": {
                "error": f"{device_name} did not produce a result artifact",
                "execution_status": execution_status,
            },
        }
    record = dict(plan or _fallback_plan(head_sha))
    results = [execution] if execution.get("component") != "runner" else []
    record.update(
        {
            "kind": "ci_execution",
            "outcome": execution["outcome"],
            "run_url": run_url,
            "attempt_id": attempt_id,
            "head_sha": str(record.get("head_sha") or head_sha),
            "results": results,
        }
    )
    return record


def _job(plan: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if plan is None:
        return {}
    jobs = plan.get("jobs", [])
    if not isinstance(jobs, list):
        return {}
    for job in jobs:
        if isinstance(job, Mapping) and job.get("mode") == "hf_checkpoint":
            return job
    return {}


def _fallback_plan(head_sha: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "approved_job_plan",
        "head_sha": head_sha,
        "outcome": "infrastructure_failure",
        "jobs": [],
        "gates": [],
        "errors": [
            {
                "code": "attempt_preparation_failed",
                "component": "planner",
                "subject": "pull_request",
            }
        ],
    }


def _load(path: Path | None) -> Mapping[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--dispatch", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--execution-status", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = _load(args.plan)
    dispatch = _load(args.dispatch)
    result = _load(args.result)
    record = report(
        plan,
        dispatch,
        result,
        args.run_url,
        args.attempt_id,
        args.head_sha,
        args.execution_status,
    )
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(BotOutput(record).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
