from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ci.bot import BotOutput
from ci.component_config import materialize
from ci.control import _refused_paths, control_record, release_control
from ci.delegator import create_delegator, diff_from_git
from ci.execution_security import canonical_digest, validate_job
from ci.hosted_checks import resolved_record, run_hosted_checks
from ci.workflow_report import report_batch


class RepositoryAdapterError(ValueError):
    pass


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory() as directory:
            configuration = Path(directory)
            materialize(args.head_checkout, args.head_sha, configuration)
            delegator = create_delegator(
                args.repository_path / "ci" / "change-rules.yaml",
                configuration,
                args.head_checkout,
            )
            diff = diff_from_git(args.base_sha, args.head_sha, args.head_checkout)
            plan = delegator.plan_context(diff.context())
        refused = _refused_paths(
            diff.changed_files,
            args.repository_path / "ci" / "protected_paths.yaml",
        )
        if refused:
            plan["blocked"].append(
                {
                    "component": "planner",
                    "reason": "protected_ci_files_changed",
                    "changed_paths": refused,
                }
            )
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
        plan = _blocked_plan(args, "invalid_ci_configuration", str(error))
    except subprocess.CalledProcessError as error:
        plan = _blocked_plan(args, "planner_failed", str(error))
    return control_record(
        plan,
        args.repository,
        args.pr_number,
        args.run_url,
        contract_sha=args.contract_sha,
    )


def _blocked_plan(args: argparse.Namespace, reason: str, detail: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_sha": args.base_sha,
        "target_sha": args.base_sha,
        "head_sha": args.head_sha,
        "rules": [],
        "components": [],
        "jobs": [],
        "gates": [],
        "checks": [],
        "blocked": [
            {
                "component": "planner",
                "reason": reason,
                "detail": detail[:500],
            }
        ],
    }


def prepare_record(args: argparse.Namespace) -> dict[str, Any]:
    planned = build_plan(args)
    if planned.get("outcome") == "blocked":
        released = dict(planned)
        released["jobs"] = []
    else:
        released = release_control(planned, args.head_sha, approve_gates=True)
    jobs = []
    devices = []
    for index, value in enumerate(released.get("jobs", [])):
        job = dict(value)
        job["repository"] = args.repository
        job.pop("manifest_digest", None)
        validate_job(job, require_digest=False)
        job["manifest_digest"] = canonical_digest(job)
        identifier = str(job["id"])
        stem = re.sub(r"[^A-Za-z0-9._-]", "-", identifier)[:100]
        filename = f"{index:03d}-{stem}.json"
        jobs.append(job)
        devices.append({"id": identifier, "file": filename, "manifest": job})
    released.update(
        {
            "attempt_id": args.attempt_id,
            "repository": args.repository,
            "base_sha": args.base_sha,
            "head_sha": args.head_sha,
            "contract_sha": args.contract_sha,
            "terminal_state": "blocked" if released.get("errors") else "planned",
            "jobs": jobs,
            "device_jobs": devices,
        }
    )
    return released


def _write(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _plan(args: argparse.Namespace) -> int:
    record = build_plan(args)
    _write(record, args.output)
    args.summary.write_text(BotOutput(record).render())
    if args.github_output is not None:
        hosted = any(
            check.get("execution_target") == "github_hosted"
            for check in record.get("checks", [])
            if isinstance(check, Mapping)
        )
        with args.github_output.open("a") as stream:
            stream.write(f"outcome={record['outcome']}\n")
            stream.write(f"has_hosted_checks={str(hosted).lower()}\n")
    return 0


def _hosted_checks(args: argparse.Namespace) -> int:
    control = _load(args.control)
    results = run_hosted_checks(
        control,
        args.head_checkout,
        args.base_sha,
        args.head_sha,
    )
    record = resolved_record(control, results)
    _write(record, args.output)
    args.summary.write_text(BotOutput(record).render())
    if args.github_output is not None:
        with args.github_output.open("a") as stream:
            stream.write(f"outcome={record['hosted_outcome']}\n")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    record = prepare_record(args)
    args.jobs.mkdir(mode=0o700)
    for job in record["device_jobs"]:
        _write(job["manifest"], args.jobs / job["file"])
    _write(record, args.output)
    return 0


def _report(args: argparse.Namespace) -> int:
    control = _load(args.control)
    if control.get("attempt_id") != args.attempt_id:
        raise RepositoryAdapterError("report attempt does not match control record")
    if control.get("head_sha") != args.head_sha:
        raise RepositoryAdapterError("report head does not match control record")
    results = _results(args.results, control)
    items = []
    for job in control.get("jobs", []):
        if not isinstance(job, Mapping):
            continue
        identifier = str(job.get("id", ""))
        items.append(
            {
                "key": identifier,
                "work": dict(job),
                "dispatch": {"job": dict(job), "outcome": "dispatching"},
            }
        )
    record = report_batch(
        control,
        {"items": items},
        results,
        args.run_url,
        args.attempt_id,
        args.head_sha,
        "success" if len(results) == len(items) else "failure",
    )
    args.output.write_text(BotOutput(record).render())
    return 0


def _results(
    directory: Path, control: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    expected = {
        str(job.get("id"))
        for job in control.get("jobs", [])
        if isinstance(job, Mapping)
    }
    results = {}
    for path in sorted(directory.rglob("*.result.json")):
        value = _load(path)
        identifier = value.get("job_id")
        if not isinstance(identifier, str) or identifier not in expected:
            raise RepositoryAdapterError("result job does not match control record")
        if value.get("repository") != control.get("repository"):
            raise RepositoryAdapterError(
                "result repository does not match control record"
            )
        results[identifier] = value
    return results


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise RepositoryAdapterError(f"{path} must contain an object")
    return value


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-path", type=Path, required=True)
    parser.add_argument("--base-checkout", type=Path, required=True)
    parser.add_argument("--head-checkout", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--run-url", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    _common(plan)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--summary", type=Path, required=True)
    plan.add_argument("--github-output", type=Path)
    plan.set_defaults(handler=_plan)

    hosted = commands.add_parser("hosted-checks")
    hosted.add_argument("--control", type=Path, required=True)
    hosted.add_argument("--repository-path", type=Path, required=True)
    hosted.add_argument("--base-checkout", type=Path, required=True)
    hosted.add_argument("--head-checkout", type=Path, required=True)
    hosted.add_argument("--base-sha", required=True)
    hosted.add_argument("--head-sha", required=True)
    hosted.add_argument("--output", type=Path, required=True)
    hosted.add_argument("--summary", type=Path, required=True)
    hosted.add_argument("--github-output", type=Path)
    hosted.set_defaults(handler=_hosted_checks)

    prepare = commands.add_parser("prepare")
    _common(prepare)
    prepare.add_argument("--attempt-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--jobs", type=Path, required=True)
    prepare.set_defaults(handler=_prepare)

    renderer = commands.add_parser("report")
    renderer.add_argument("--control", type=Path, required=True)
    renderer.add_argument("--results", type=Path, required=True)
    renderer.add_argument("--run-url", required=True)
    renderer.add_argument("--attempt-id", required=True)
    renderer.add_argument("--head-sha", required=True)
    renderer.add_argument("--output", type=Path, required=True)
    renderer.set_defaults(handler=_report)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
