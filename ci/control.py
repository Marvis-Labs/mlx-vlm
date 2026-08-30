from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ci.bot import BotOutput
from ci.delegator import create_delegator, diff_from_git


class PlanningOutcome(str, Enum):
    BLOCKED = "blocked"
    AWAITING_APPROVAL = "awaiting_approval"
    READY = "ready"


class ExecutionOutcome(str, Enum):
    PASSED = "passed"
    TEST_FAILURE = "test_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    CANCELLED = "cancelled"


class ControlError(ValueError):
    pass


def configuration_digest(configuration: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(configuration)).hexdigest()


def plan_digest(plan: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(plan)).hexdigest()


def planning_outcome(plan: Mapping[str, Any]) -> PlanningOutcome:
    if plan.get("blocked"):
        return PlanningOutcome.BLOCKED
    if plan.get("gates"):
        return PlanningOutcome.AWAITING_APPROVAL
    return PlanningOutcome.READY


def execution_outcome(
    exit_code: int | None,
    *,
    cancelled: bool = False,
    infrastructure_failure: bool = False,
) -> ExecutionOutcome:
    if cancelled:
        return ExecutionOutcome.CANCELLED
    if infrastructure_failure or exit_code is None:
        return ExecutionOutcome.INFRASTRUCTURE_FAILURE
    if exit_code == 0:
        return ExecutionOutcome.PASSED
    return ExecutionOutcome.TEST_FAILURE


def control_record(
    plan: Mapping[str, Any],
    repository: str,
    pr_number: int,
    run_url: str | None = None,
) -> dict[str, Any]:
    _validate_plan(plan)
    digest = plan_digest(plan)
    head_sha = plan["head_sha"]
    outcome = planning_outcome(plan)
    return {
        "schema_version": 1,
        "kind": "ci_control",
        "run_id": (f"{repository}:pull:{pr_number}:{head_sha[:12]}:{digest[7:19]}"),
        "repository": repository,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "plan_digest": digest,
        "outcome": outcome.value,
        "run_url": run_url,
        "rules": plan["rules"],
        "components": plan["components"],
        "checks": plan["checks"],
        "jobs": plan["jobs"],
        "gates": plan["gates"],
        "errors": [_error_record(item) for item in plan["blocked"]],
    }


def release_control(
    control: Mapping[str, Any],
    current_head_sha: str,
    *,
    approve_gates: bool = False,
) -> dict[str, Any]:
    _validate_control(control)
    if control["outcome"] == PlanningOutcome.BLOCKED.value:
        raise ControlError("a blocked plan cannot be released")
    if control["head_sha"] != current_head_sha:
        raise ControlError("the pull request head changed before approval")
    if control["gates"] and not approve_gates:
        raise ControlError("maintainer approval gates have not been released")

    jobs = list(control["jobs"])
    resolved_gates: list[dict[str, Any]] = []
    gate_ids: list[str] = []
    for gate in control["gates"]:
        _validate_gate(gate, current_head_sha)
        jobs.append(gate["pending_work"])
        gate_ids.append(gate["id"])
        resolved_gates.append(
            {
                key: value
                for key, value in gate.items()
                if key not in {"configuration", "pending_work"}
            }
            | {"status": "approved"}
        )
    _require_unique_job_ids(jobs)

    released = dict(control)
    released.update(
        {
            "kind": "approved_job_plan",
            "outcome": PlanningOutcome.READY.value,
            "jobs": jobs,
            "gates": resolved_gates,
            "approval": {
                "mechanism": ("github_environment" if gate_ids else "not_required"),
                "head_sha": current_head_sha,
                "gate_ids": gate_ids,
            },
        }
    )
    released["job_plan_digest"] = plan_digest(
        {
            "head_sha": current_head_sha,
            "jobs": jobs,
            "approval": released["approval"],
        }
    )
    return released


def render_status(record: Mapping[str, Any]) -> str:
    return BotOutput(record).render()


def _error_record(blocker: Mapping[str, Any]) -> dict[str, Any]:
    code = str(blocker.get("reason", "planner_failed"))
    user_configuration = code.startswith(("missing_", "invalid_")) or code in {
        "not_configured",
        "unknown_scenario",
        "model_manifest_not_updated",
    }
    return {
        "code": code,
        "category": "configuration" if user_configuration else "planning",
        "component": str(blocker.get("component", "planner")),
        "subject": str(blocker.get("model") or blocker.get("rule") or "pull_request"),
        "retryable": code == "planner_failed",
        "user_actionable": user_configuration,
        "details": {
            key: value
            for key, value in blocker.items()
            if key not in {"reason", "component", "model", "rule"}
        },
    }


def _validate_plan(plan: Mapping[str, Any]) -> None:
    required_lists = ("rules", "components", "checks", "jobs", "gates", "blocked")
    if plan.get("schema_version") != 1:
        raise ControlError("unsupported plan schema_version")
    if not isinstance(plan.get("head_sha"), str) or not plan["head_sha"]:
        raise ControlError("plan requires an immutable head_sha")
    if any(not isinstance(plan.get(key), list) for key in required_lists):
        raise ControlError("plan collections must be lists")
    _require_unique_job_ids(plan["jobs"])
    _require_unique_check_ids(plan["checks"])


def _validate_control(control: Mapping[str, Any]) -> None:
    if control.get("schema_version") != 1 or control.get("kind") != "ci_control":
        raise ControlError("invalid control record")
    if control.get("outcome") not in {outcome.value for outcome in PlanningOutcome}:
        raise ControlError("invalid planning outcome")


def _validate_gate(gate: Mapping[str, Any], current_head_sha: str) -> None:
    if gate.get("type") != "maintainer_approval":
        raise ControlError("unsupported approval gate")
    if gate.get("status") != "awaiting_maintainer_approval":
        raise ControlError("approval gate is not awaiting approval")
    if gate.get("head_sha") != current_head_sha:
        raise ControlError("approval gate does not match the current head")
    configuration = gate.get("configuration")
    if not isinstance(configuration, dict):
        raise ControlError("approval gate has no configuration")
    if gate.get("configuration_digest") != configuration_digest(configuration):
        raise ControlError("approval gate configuration digest does not match")
    pending_work = gate.get("pending_work")
    if not isinstance(pending_work, Mapping):
        raise ControlError("approval gate has no pending work")
    requested = gate.get("requested_phases")
    if not isinstance(requested, list) or not requested:
        raise ControlError("approval gate has no requested phases")
    if (
        pending_work.get("work_type") != "ModelPath"
        or pending_work.get("component") != "model_path"
        or pending_work.get("model") != gate.get("model")
        or pending_work.get("phases") != requested
    ):
        raise ControlError("approval gate pending work exceeds its scope")


def _require_unique_job_ids(jobs: Sequence[Mapping[str, Any]]) -> None:
    identifiers = [job.get("id") for job in jobs]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise ControlError("every job requires an id")
    if len(identifiers) != len(set(identifiers)):
        raise ControlError("job ids must be unique")


def _require_unique_check_ids(checks: Sequence[Mapping[str, Any]]) -> None:
    identifiers = [check.get("id") for check in checks]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise ControlError("every hosted check requires an id")
    if len(identifiers) != len(set(identifiers)):
        raise ControlError("hosted check ids must be unique")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _refused_paths(paths: Sequence[str], config: Path) -> list[str]:
    data = yaml.safe_load(config.read_text())
    if not isinstance(data, Mapping) or not isinstance(data.get("refuse"), list):
        raise ControlError("protected path configuration is invalid")
    patterns = [str(pattern).split(" #", 1)[0].rstrip() for pattern in data["refuse"]]
    return sorted(
        path
        for path in paths
        if any(
            path == pattern or (pattern.endswith("/") and path.startswith(pattern))
            for pattern in patterns
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ControlError(f"{path} must contain a JSON object")
    return data


def _write_record(
    record: Mapping[str, Any], output: Path, markdown: Path, github_output: Path | None
) -> None:
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    markdown.write_text(render_status(record))
    if github_output:
        with github_output.open("a") as stream:
            stream.write(f"outcome={record['outcome']}\n")
            stream.write(f"has_gates={str(bool(record['gates'])).lower()}\n")
            stream.write(f"has_jobs={str(bool(record['jobs'])).lower()}\n")
            stream.write(f"has_hosted_checks={str(bool(record['checks'])).lower()}\n")


def _blocked_plan(head_sha: str, reason: str, detail: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "head_sha": head_sha,
        "rules": [],
        "components": [],
        "checks": [],
        "jobs": [],
        "gates": [],
        "blocked": [
            {
                "component": "planner",
                "reason": reason,
                "detail": detail[:500],
            }
        ],
    }


def _plan_command(args: argparse.Namespace) -> int:
    try:
        delegator = create_delegator(
            args.rules_config, args.model_config, args.scenario_config
        )
        diff = diff_from_git(args.base, args.head, args.repository_path)
        plan = delegator.plan_context(diff.context())
        refused = _refused_paths(diff.changed_files, args.protected_config)
        if refused:
            plan["blocked"].append(
                {
                    "component": "planner",
                    "reason": "protected_ci_files_changed",
                    "changed_paths": refused,
                }
            )
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
        plan = _blocked_plan(args.head, "invalid_ci_configuration", str(error))
    except subprocess.CalledProcessError as error:
        plan = _blocked_plan(args.head, "planner_failed", str(error))
    record = control_record(plan, args.repository, args.pr, args.run_url)
    _write_record(record, args.output, args.markdown, args.github_output)
    return 0


def _release_command(args: argparse.Namespace) -> int:
    released = release_control(
        _load_json(args.control),
        args.current_head,
        approve_gates=args.approve_gates,
    )
    _write_record(released, args.output, args.markdown, args.github_output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--base", required=True)
    plan_parser.add_argument("--head", required=True)
    plan_parser.add_argument("--repository", required=True)
    plan_parser.add_argument("--repository-path", type=Path, default=Path.cwd())
    plan_parser.add_argument("--pr", type=int, required=True)
    plan_parser.add_argument("--rules-config", type=Path, required=True)
    plan_parser.add_argument("--model-config", type=Path, required=True)
    plan_parser.add_argument("--scenario-config", type=Path, required=True)
    plan_parser.add_argument("--protected-config", type=Path, required=True)
    plan_parser.add_argument("--run-url")
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--markdown", type=Path, required=True)
    plan_parser.add_argument("--github-output", type=Path)
    plan_parser.set_defaults(handler=_plan_command)

    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--control", type=Path, required=True)
    release_parser.add_argument("--current-head", required=True)
    release_parser.add_argument("--output", type=Path, required=True)
    release_parser.add_argument("--markdown", type=Path, required=True)
    release_parser.add_argument("--github-output", type=Path)
    release_parser.add_argument("--approve-gates", action="store_true")
    release_parser.set_defaults(handler=_release_command)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
