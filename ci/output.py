from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


class OutputError(ValueError):
    pass


SHA = re.compile(r"[0-9a-f]{40}\Z")


COMPONENT_LABELS = {
    "model_path": "ModelPath",
    "server_change": "ServerChange",
}
STATUS_LABELS = {
    "passed": "Passed",
    "failed": "Failed",
    "skipped": "Skipped",
    "infrastructure_failure": "Infrastructure failure",
}
BLOCK_REASONS = {
    "model_case_missing": "No synthetic model case is registered for this family.",
}


def _text(value: Any) -> str:
    return (
        html.escape(str(value), quote=False).replace("|", "\\|").replace("@", "@\u200b")
    )


def _paths(attempt: Mapping[str, Any], job: Mapping[str, Any]) -> list[str]:
    changed = attempt.get("changed_files", [])
    if job["component"] == "model_path":
        prefix = f"mlx_vlm/models/{job['subject']}/"
        return [path for path in changed if str(path).startswith(prefix)]
    if job["component"] == "server_change":
        return [
            path
            for path in changed
            if str(path).startswith("mlx_vlm/server/")
            or path == "mlx_vlm/tests/test_server.py"
        ]
    return []


def _section_status(result: Mapping[str, Any] | None) -> str:
    if result is None:
        return "Pending"
    status = str(result.get("status", ""))
    if status != "failed":
        return STATUS_LABELS.get(status, "Failed")
    failed = {
        check.get("category")
        for check in result.get("checks", [])
        if check.get("status") == "failed"
    }
    if "correctness" in failed:
        return "Correctness failed"
    if "performance" in failed:
        return "Performance regressed"
    return "Failed"


def _measurement(value: Any, unit: str) -> str:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise OutputError("metric value is invalid")
    rendered = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}".strip()


def _change(value: Any) -> str:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise OutputError("metric change is invalid")
    rendered = f"{value:+.2f}%"
    return f"**{rendered}**" if abs(value) >= 4 else rendered


def _section(
    attempt: Mapping[str, Any],
    job: Mapping[str, Any],
    result: Mapping[str, Any] | None,
) -> list[str]:
    status = _section_status(result)
    details = "<details>" if status == "Passed" else "<details open>"
    lines = [
        details,
        f"<summary><strong>{_text(job['subject'])}</strong> · "
        f"{_text(COMPONENT_LABELS[job['component']])} · {_text(status)}</summary>",
        "",
    ]
    paths = _paths(attempt, job)
    if paths:
        shown = ", ".join(f"`{_text(path)}`" for path in paths[:8])
        if len(paths) > 8:
            shown += f", and {len(paths) - 8} more"
        lines.extend([f"Changed: {shown}", ""])
    if result is None:
        lines.extend(["Result has not been reported.", "", "</details>"])
        return lines
    device = result["device"]
    if device is None:
        lines.append("Device: unavailable  ")
    else:
        lines.append(
            f"Device: {_text(device['chip'])} · {_text(device['memory_gib'])} GB unified memory  "
        )
    artifact = job.get("artifact")
    cache = str(result["cache"]).replace("_", " ").capitalize()
    if artifact:
        revision = str(artifact["revision"])[:8]
        lines.append(
            f"Checkpoint: {_text(cache)} · `{_text(artifact['repository'])}@{revision}`  "
        )
    lines.extend([f"Duration: {_measurement(result['duration_ms'], 'ms')}", ""])
    checks = result.get("checks", [])
    if checks:
        lines.extend(["| Validation | Result | Details |", "|---|---|---|"])
        for check in checks:
            lines.append(
                f"| {_text(check['name'])} | {_text(STATUS_LABELS[check['status']])} "
                f"| {_text(check['detail'])} |"
            )
    metrics = result.get("metrics", [])
    if metrics:
        lines.extend(
            [
                "",
                "| Metric | Main | PR | Change | Verdict |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for metric in metrics:
            lines.append(
                f"| {_text(metric['name'])} | "
                f"{_measurement(metric['base'], metric['unit'])} | "
                f"{_measurement(metric['head'], metric['unit'])} | "
                f"{_change(metric['change_pct'])} | "
                f"{_text(metric['verdict'].capitalize())} |"
            )
    lines.extend(["", "</details>"])
    return lines


def _blocked_section(blocked: Mapping[str, Any]) -> list[str]:
    component = str(blocked.get("component", ""))
    subject = str(blocked.get("subject", ""))
    reason = BLOCK_REASONS.get(
        str(blocked.get("reason", "")), "This CI section is not configured."
    )
    return [
        "<details open>",
        f"<summary><strong>{_text(subject)}</strong> · "
        f"{_text(COMPONENT_LABELS.get(component, component))} · Blocked</summary>",
        "",
        _text(reason),
        "",
        "</details>",
    ]


def render_comment(
    attempt: Mapping[str, Any],
    jobs_document: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    run_url: str,
) -> str:
    jobs = jobs_document.get("jobs", [])
    blocked = jobs_document.get("blocked", [])
    by_id = {result.get("job_id"): result for result in results}
    if len(by_id) != len(results):
        raise OutputError("result identifiers are duplicated")
    statuses = [_section_status(by_id.get(job["id"])) for job in jobs]
    passed = statuses.count("Passed")
    total = len(jobs) + len(blocked)
    if blocked:
        overall = "Blocked"
    elif jobs and passed == len(jobs):
        overall = "Passed"
    elif any(status == "Infrastructure failure" for status in statuses):
        overall = "Infrastructure failure"
    elif any(status not in {"Passed", "Pending"} for status in statuses):
        overall = "Failed"
    else:
        overall = "Pending"
    attempt_id = f"{attempt['run_id']}.{attempt['run_attempt']}"
    lines = [
        f"<!-- mixie:attempt:{attempt_id} -->",
        "### Mixie",
        "",
        f"{overall} — {passed} of {total} sections passed",
        "",
        f"PR `{str(attempt['head_sha'])[:8]}` against main "
        f"`{str(attempt['base_sha'])[:8]}` · Attempt `{attempt_id}` · "
        f"[Workflow run]({_text(run_url)})",
    ]
    for job in jobs:
        lines.extend(["", *_section(attempt, job, by_id.get(job["id"]))])
    for item in blocked:
        lines.extend(["", *_blocked_section(item)])
    return "\n".join(lines) + "\n"


def validate_dispatch(event: Mapping[str, Any]) -> tuple[int, int]:
    if (
        not {"action", "client_payload"}.issubset(event)
        or event.get("action") != "ci-run-result"
    ):
        raise OutputError("unsupported result event")
    payload = event.get("client_payload")
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "run_id",
        "run_attempt",
    }:
        raise OutputError("result event fields are invalid")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise OutputError("unsupported result event version")
    run_id, run_attempt = payload["run_id"], payload["run_attempt"]
    if (
        type(run_id) is not int
        or not 1 <= run_id <= 10**18
        or type(run_attempt) is not int
        or not 1 <= run_attempt <= 1_000
    ):
        raise OutputError("result run identity is invalid")
    return run_id, run_attempt


def render_coalesced(event: Mapping[str, Any]) -> tuple[int, str]:
    if event.get("action") != "ci-run-coalesced":
        raise OutputError("unsupported coalesced event")
    payload = event.get("client_payload")
    expected = {
        "schema_version",
        "pull_request",
        "comment_id",
        "requested_at",
        "base_sha",
        "head_sha",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise OutputError("coalesced event fields are invalid")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise OutputError("unsupported coalesced event version")
    pull_request, comment_id = payload["pull_request"], payload["comment_id"]
    if (
        type(pull_request) is not int
        or not 1 <= pull_request <= 1_000_000
        or type(comment_id) is not int
        or not 1 <= comment_id <= 10**18
        or SHA.fullmatch(str(payload["base_sha"])) is None
        or SHA.fullmatch(str(payload["head_sha"])) is None
    ):
        raise OutputError("coalesced event identity is invalid")
    requested_at = payload["requested_at"]
    try:
        parsed = datetime.fromisoformat(str(requested_at).replace("Z", "+00:00"))
    except ValueError as error:
        raise OutputError("coalesced request time is invalid") from error
    if parsed.tzinfo is None:
        raise OutputError("coalesced request time is invalid")
    body = "\n".join(
        [
            f"<!-- mixie:coalesced:{comment_id} -->",
            "### Mixie",
            "",
            "This request joined the active CI attempt for "
            f"PR `{str(payload['head_sha'])[:8]}` against main "
            f"`{str(payload['base_sha'])[:8]}`.",
            "",
            f"Requested at `{_text(requested_at)}`. No duplicate runner work was started.",
            "",
        ]
    )
    return pull_request, body


def validate_bundle(
    bundle: Mapping[str, Any], repository: str, run_id: int, run_attempt: int
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]], str]:
    if not isinstance(bundle, Mapping) or set(bundle) != {
        "schema_version",
        "attempt",
        "jobs",
        "results",
        "run_url",
    }:
        raise OutputError("result bundle fields are invalid")
    if bundle["schema_version"] != 1 or type(bundle["schema_version"]) is not int:
        raise OutputError("unsupported result bundle version")
    attempt, jobs_document, results = (
        bundle["attempt"],
        bundle["jobs"],
        bundle["results"],
    )
    if not isinstance(attempt, Mapping) or not isinstance(jobs_document, Mapping):
        raise OutputError("result bundle identity is invalid")
    if (
        attempt.get("repository") != repository
        or attempt.get("run_id") != run_id
        or attempt.get("run_attempt") != run_attempt
        or not isinstance(attempt.get("pull_request"), int)
        or SHA.fullmatch(str(attempt.get("base_sha", ""))) is None
        or SHA.fullmatch(str(attempt.get("head_sha", ""))) is None
    ):
        raise OutputError("result bundle does not match this run")
    jobs = jobs_document.get("jobs")
    if not isinstance(jobs, list) or not isinstance(results, list):
        raise OutputError("result bundle work is invalid")
    by_id = {
        result.get("job_id"): result
        for result in results
        if isinstance(result, Mapping)
    }
    if len(by_id) != len(results) or len(results) != len(jobs):
        raise OutputError("result bundle is incomplete")
    for job in jobs:
        if not isinstance(job, Mapping):
            raise OutputError("result bundle job is invalid")
        result = by_id.get(job.get("id"))
        if (
            result is None
            or result.get("manifest_digest") != job.get("manifest_digest")
            or job.get("repository") != repository
            or job.get("base_sha") != attempt["base_sha"]
            or job.get("head_sha") != attempt["head_sha"]
        ):
            raise OutputError("result is not bound to its sealed job")
    run_url = bundle["run_url"]
    expected = f"https://github.com/Marvis-Labs/mlx-ci/actions/runs/{run_id}"
    if run_url != expected:
        raise OutputError("result run URL is invalid")
    return attempt, jobs_document, results, run_url


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise OutputError("input must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OutputError("input is invalid JSON") from error
    if not isinstance(value, dict):
        raise OutputError("input must be an object")
    return value


def _write(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    dispatch = subparsers.add_parser("dispatch")
    dispatch.add_argument("--event", required=True, type=Path)
    dispatch.add_argument("--github-output", required=True, type=Path)
    comment = subparsers.add_parser("comment")
    comment.add_argument("--bundle", required=True, type=Path)
    comment.add_argument("--repository", required=True)
    comment.add_argument("--run-id", required=True, type=int)
    comment.add_argument("--run-attempt", required=True, type=int)
    comment.add_argument("--output", required=True, type=Path)
    comment.add_argument("--request", required=True, type=Path)
    comment.add_argument("--github-output", required=True, type=Path)
    coalesced = subparsers.add_parser("coalesced")
    coalesced.add_argument("--event", required=True, type=Path)
    coalesced.add_argument("--request", required=True, type=Path)
    coalesced.add_argument("--github-output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.command == "dispatch":
        run_id, run_attempt = validate_dispatch(_read(arguments.event))
        with arguments.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"run_id={run_id}\nrun_attempt={run_attempt}\n")
        return 0
    if arguments.command == "coalesced":
        pull_request, body = render_coalesced(_read(arguments.event))
        _write(arguments.request, json.dumps({"body": body}, ensure_ascii=False) + "\n")
        with arguments.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"pull_request={pull_request}\n")
        return 0
    attempt, jobs, results, run_url = validate_bundle(
        _read(arguments.bundle),
        arguments.repository,
        arguments.run_id,
        arguments.run_attempt,
    )
    body = render_comment(attempt, jobs, results, run_url)
    _write(arguments.output, body)
    _write(arguments.request, json.dumps({"body": body}, ensure_ascii=False) + "\n")
    with arguments.github_output.open("a", encoding="utf-8") as stream:
        stream.write(f"pull_request={attempt['pull_request']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
