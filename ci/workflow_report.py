from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.bot import BotOutput


def report(
    plan: Mapping[str, Any] | None,
    dispatch: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    run_url: str,
    attempt_id: str,
    head_sha: str,
    execution_status: str,
) -> dict[str, Any]:
    if (
        isinstance(plan, Mapping)
        and plan.get("outcome") in {"blocked", "awaiting_approval"}
        and dispatch is None
        and result is None
    ):
        return _record(plan, [], run_url, attempt_id, head_sha, [])
    execution = _execution(plan, dispatch, result, execution_status)
    return _record(
        plan,
        [execution],
        run_url,
        attempt_id,
        head_sha,
        [dispatch.get("lease")] if dispatch is not None else [],
    )


def report_batch(
    plan: Mapping[str, Any] | None,
    batch: Mapping[str, Any] | None,
    results: Mapping[str, Mapping[str, Any]],
    run_url: str,
    attempt_id: str,
    head_sha: str,
    execution_status: str,
) -> dict[str, Any]:
    if batch is None:
        return _record(
            plan,
            [],
            run_url,
            attempt_id,
            head_sha,
            [],
        )
    executions: list[dict[str, Any]] = []
    leases: list[Any] = []
    for item in batch.get("items", []):
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key", ""))
        dispatch = item.get("dispatch")
        work = item.get("work")
        dispatch_value = dict(dispatch) if isinstance(dispatch, Mapping) else None
        plan_value = {"jobs": [work]} if isinstance(work, Mapping) else plan
        executions.append(
            _execution(
                plan_value,
                dispatch_value,
                results.get(key),
                execution_status,
            )
        )
        if isinstance(item.get("lease"), Mapping):
            leases.append(item["lease"])
    return _record(
        plan,
        executions,
        run_url,
        attempt_id,
        head_sha,
        leases,
    )


def report_coalesced(
    plan: Mapping[str, Any] | None,
    run_url: str,
    attempt_id: str,
    head_sha: str,
    owner_attempt_id: str,
    owner_run_url: str,
) -> dict[str, Any]:
    jobs = plan.get("jobs", []) if isinstance(plan, Mapping) else []
    executions = [
        {
            "component": str(job.get("component", "runner")),
            "model": job.get("model"),
            "profile": job.get("profile"),
            "job_id": str(job.get("id", "")),
            "outcome": "coalesced",
            "owner_attempt_id": owner_attempt_id,
            "owner_run_url": owner_run_url,
        }
        for job in jobs
        if isinstance(job, Mapping)
    ]
    record = _record(
        plan,
        executions,
        run_url,
        attempt_id,
        head_sha,
        [],
    )
    record["outcome"] = "coalesced"
    record["coalesced_with"] = {
        "attempt_id": owner_attempt_id,
        "run_url": owner_run_url,
    }
    return record


def _execution(
    plan: Mapping[str, Any] | None,
    dispatch: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    execution_status: str,
) -> dict[str, Any]:
    if dispatch is not None and dispatch.get("outcome") == "no_eligible_runner":
        if result is None or result.get("decision") != "accepted":
            return _no_runner_result(dispatch)
    if result is not None and dispatch is not None:
        lease = dispatch.get("lease")
        if (
            isinstance(lease, Mapping)
            and result.get("device")
            and result.get("device") != lease.get("device")
        ):
            result = None
    if result is not None:
        execution = dict(result)
        if dispatch is not None:
            execution.setdefault("attempts", list(dispatch.get("attempts", [])))
            execution.setdefault("unavailable", list(dispatch.get("unavailable", [])))
        return execution
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
    return {
        "component": str(job.get("component", "runner")),
        "model": job.get("model"),
        "profile": job.get("profile"),
        "job_id": str(job.get("id", "")),
        "outcome": outcome,
        "selected_device": selected_device,
        "findings": {
            "error": f"{device_name} did not produce a result artifact",
            "execution_status": execution_status,
        },
    }


def _no_runner_result(dispatch: Mapping[str, Any]) -> dict[str, Any]:
    job = dispatch.get("job", {})
    return {
        "component": str(job.get("component", "runner")),
        "model": job.get("model"),
        "profile": job.get("profile"),
        "job_id": str(job.get("id", "")),
        "outcome": "no_eligible_runner",
        "required_memory_gib": dispatch.get("required_memory_gib"),
        "required_disk_gib": dispatch.get("required_disk_gib"),
        "attempts": list(dispatch.get("attempts", [])),
        "unavailable": list(dispatch.get("unavailable", [])),
        "selected_device": None,
    }


def _record(
    plan: Mapping[str, Any] | None,
    executions: Sequence[Mapping[str, Any]],
    run_url: str,
    attempt_id: str,
    head_sha: str,
    leases: Sequence[Any],
) -> dict[str, Any]:
    record = dict(plan or _fallback_plan(head_sha))
    results = [
        dict(execution)
        for execution in executions
        if execution.get("component") != "runner"
    ]
    outcomes = {str(result.get("outcome", "")) for result in results}
    planning_outcome = str(record.get("outcome", ""))
    fallback_outcome = (
        planning_outcome
        if planning_outcome in {"blocked", "awaiting_approval"}
        else "infrastructure_failure" if record.get("errors") else "passed"
    )
    outcome = next(
        (
            value
            for value in (
                "test_failure",
                "regressed",
                "no_eligible_runner",
                "infrastructure_failure",
                "cancelled",
                "improved",
                "passed",
            )
            if value in outcomes
        ),
        fallback_outcome,
    )
    record.update(
        {
            "kind": "ci_execution",
            "outcome": outcome,
            "run_url": run_url,
            "attempt_id": attempt_id,
            "head_sha": str(record.get("head_sha") or head_sha),
            "results": results,
        }
    )
    record["device_leases"] = [
        dict(lease) for lease in leases if isinstance(lease, Mapping)
    ]
    return record


def _job(plan: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if plan is None:
        return {}
    jobs = plan.get("jobs", [])
    if not isinstance(jobs, list):
        return {}
    for job in jobs:
        if isinstance(job, Mapping):
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
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--results-directory", type=Path)
    parser.add_argument("--coalesced-owner")
    parser.add_argument("--coalesced-url")
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--execution-status", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = _load(args.plan)
    if args.coalesced_owner:
        record = report_coalesced(
            plan,
            args.run_url,
            args.attempt_id,
            args.head_sha,
            args.coalesced_owner,
            args.coalesced_url or "",
        )
    elif args.batch:
        batch = _load(args.batch)
        results = _load_results(args.results_directory)
        record = report_batch(
            plan,
            batch,
            results,
            args.run_url,
            args.attempt_id,
            args.head_sha,
            args.execution_status,
        )
    else:
        record = report(
            plan,
            _load(args.dispatch),
            _load(args.result),
            args.run_url,
            args.attempt_id,
            args.head_sha,
            args.execution_status,
        )
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(BotOutput(record).render())
    return 0


def _load_results(directory: Path | None) -> dict[str, Mapping[str, Any]]:
    if directory is None or not directory.is_dir():
        return {}
    results: dict[str, Mapping[str, Any]] = {}
    for path in sorted(directory.rglob("result-*.json")):
        value = _load(path)
        if value is not None:
            results[path.stem.removeprefix("result-")] = value
    return results


if __name__ == "__main__":
    raise SystemExit(main())
