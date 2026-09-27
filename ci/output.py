from __future__ import annotations

import html
import math
from typing import Any, Mapping, Sequence


class OutputError(ValueError):
    pass


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


def render_comment(
    attempt: Mapping[str, Any],
    jobs_document: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    run_url: str,
) -> str:
    jobs = jobs_document.get("jobs", [])
    by_id = {result.get("job_id"): result for result in results}
    if len(by_id) != len(results):
        raise OutputError("result identifiers are duplicated")
    statuses = [_section_status(by_id.get(job["id"])) for job in jobs]
    passed = statuses.count("Passed")
    if jobs and passed == len(jobs):
        overall = "Passed"
    elif any(status == "Infrastructure failure" for status in statuses):
        overall = "Infrastructure failure"
    elif any(status not in {"Passed", "Pending"} for status in statuses):
        overall = "Failed"
    else:
        overall = "Pending"
    attempt_id = f"{attempt['run_id']}.{attempt['run_attempt']}"
    lines = [
        f"<!-- mlx-ci:attempt:{attempt_id} -->",
        f"{overall} — {passed} of {len(jobs)} sections passed",
        "",
        f"PR `{str(attempt['head_sha'])[:8]}` against main "
        f"`{str(attempt['base_sha'])[:8]}` · Attempt `{attempt_id}` · "
        f"[Workflow run]({_text(run_url)})",
    ]
    for job in jobs:
        lines.extend(["", *_section(attempt, job, by_id.get(job["id"]))])
    return "\n".join(lines) + "\n"
