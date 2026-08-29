from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class BotOutputError(ValueError):
    pass


class ComponentOutput(Protocol):
    component_names: frozenset[str]

    def sections(self, record: Mapping[str, Any]) -> Sequence["BotSection"]: ...


@dataclass(frozen=True)
class BotStage:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class BotMetric:
    check: str
    name: str
    base: Any
    head: Any
    change_pct: Any
    verdict: str
    unit: str = ""


@dataclass(frozen=True)
class BotSection:
    title: str
    component: str
    status: str
    paths: tuple[str, ...]
    stages: tuple[BotStage, ...]
    metrics: tuple[BotMetric, ...]
    messages: tuple[str, ...]

    def render(self) -> list[str]:
        lines = [
            "<details open>",
            (
                f"<summary><strong>{_cell(self.title)}</strong> · "
                f"{_cell(self.component)} · {_cell(self.status)}</summary>"
            ),
            "",
        ]
        if self.paths:
            paths = ", ".join(f"`{_cell(path)}`" for path in self.paths[:8])
            if len(self.paths) > 8:
                paths += f", and {len(self.paths) - 8} more"
            lines.extend([f"Changed: {paths}", ""])
        if self.stages:
            lines.extend(["| Check | Status | Configuration |", "|---|---|---|"])
            lines.extend(
                f"| {_cell(stage.name)} | {_cell(stage.status)} | {_cell(stage.detail)} |"
                for stage in self.stages
            )
        if self.metrics:
            lines.extend(
                [
                    "",
                    "| Check | Metric | Main | PR | Change | Verdict |",
                    "|---|---|---:|---:|---:|---|",
                ]
            )
            lines.extend(
                "| "
                + " | ".join(
                    (
                        _cell(metric.check),
                        _cell(metric.name),
                        _cell(_measurement(metric.base, metric.unit)),
                        _cell(_measurement(metric.head, metric.unit)),
                        _cell(_change(metric.change_pct)),
                        _cell(metric.verdict),
                    )
                )
                + " |"
                for metric in self.metrics
            )
        for message in self.messages:
            lines.extend(["", _cell(message)])
        lines.extend(["", "</details>"])
        return lines


class ModelPathOutput:
    """Render one independent bot section for every touched model family."""

    component_names = frozenset({"model_path", "new_model_path"})
    stage_order = ("synthetic", "hf_checkpoint")

    def sections(self, record: Mapping[str, Any]) -> Sequence[BotSection]:
        models = self._models(record)
        return tuple(self._section(record, model) for model in sorted(models))

    def _models(self, record: Mapping[str, Any]) -> set[str]:
        models: set[str] = set()
        for key in ("jobs", "gates", "results"):
            for item in _items(record, key):
                if item.get("component") in self.component_names and item.get("model"):
                    models.add(str(item["model"]))
        for error in _items(record, "errors"):
            if error.get("component") in self.component_names and error.get("subject"):
                models.add(str(error["subject"]))
        return models

    def _section(self, record: Mapping[str, Any], model: str) -> BotSection:
        jobs = self._jobs(record, model)
        gates = self._matching(record, "gates", model)
        errors = self._matching_errors(record, model)
        results = self._matching(record, "results", model)
        paths = self._paths(jobs, gates, errors)
        status = self._status(record, jobs, gates, errors, results)
        stages = self._stages(jobs, gates, errors, results)
        metrics = self._metrics(results)
        messages = self._messages(gates, errors, results)
        return BotSection(
            title=model,
            component="ModelPath",
            status=status,
            paths=paths,
            stages=stages,
            metrics=metrics,
            messages=messages,
        )

    def _matching(
        self, record: Mapping[str, Any], key: str, model: str
    ) -> list[Mapping[str, Any]]:
        return [
            item
            for item in _items(record, key)
            if item.get("component") in self.component_names
            and str(item.get("model")) == model
        ]

    def _matching_errors(
        self, record: Mapping[str, Any], model: str
    ) -> list[Mapping[str, Any]]:
        return [
            error
            for error in _items(record, "errors")
            if error.get("component") in self.component_names
            and str(error.get("subject")) == model
        ]

    def _jobs(self, record: Mapping[str, Any], model: str) -> list[Mapping[str, Any]]:
        jobs = self._matching(record, "jobs", model)
        for gate in self._matching(record, "gates", model):
            jobs.extend(
                job for job in gate.get("pending_jobs", []) if isinstance(job, Mapping)
            )
        unique: dict[str, Mapping[str, Any]] = {}
        for job in jobs:
            identifier = str(job.get("id", f"{model}:{job.get('mode', 'default')}"))
            unique[identifier] = job
        return list(unique.values())

    def _paths(
        self,
        jobs: Sequence[Mapping[str, Any]],
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        paths: set[str] = set()
        for item in (*jobs, *gates):
            paths.update(str(path) for path in item.get("changed_paths", []))
        for error in errors:
            details = error.get("details", {})
            if isinstance(details, Mapping):
                paths.update(str(path) for path in details.get("changed_paths", []))
        return tuple(sorted(paths))

    def _status(
        self,
        record: Mapping[str, Any],
        jobs: Sequence[Mapping[str, Any]],
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
    ) -> str:
        if errors:
            return "Blocked"
        outcomes = {str(result.get("outcome", "")) for result in results}
        for outcome in (
            "test_failure",
            "regressed",
            "no_eligible_runner",
            "infrastructure_failure",
            "cancelled",
            "running",
            "queued",
            "passed",
        ):
            if outcome in outcomes:
                return _label(outcome)
        if any(gate.get("status") == "awaiting_maintainer_approval" for gate in gates):
            return "Awaiting maintainer approval"
        if record.get("kind") == "approved_job_plan" and jobs:
            return "Ready for runner dispatch"
        if jobs:
            return "Awaiting /ci run"
        return "No jobs"

    def _stages(
        self,
        jobs: Sequence[Mapping[str, Any]],
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[BotStage, ...]:
        jobs_by_mode = {str(job.get("mode", "default")): job for job in jobs}
        results_by_mode = {
            str(result.get("mode", "default")): result for result in results
        }
        error_modes = {
            str(error.get("details", {}).get("mode"))
            for error in errors
            if isinstance(error.get("details"), Mapping)
            and error.get("details", {}).get("mode")
        }
        modes = set(jobs_by_mode) | set(results_by_mode) | error_modes
        ordered = [mode for mode in self.stage_order if mode in modes]
        ordered.extend(sorted(modes - set(ordered)))
        awaiting = any(
            gate.get("status") == "awaiting_maintainer_approval" for gate in gates
        )
        stages: list[BotStage] = []
        for mode in ordered:
            result = results_by_mode.get(mode)
            if mode in error_modes:
                status = "Blocked"
            elif result:
                status = _label(str(result.get("outcome", "unknown")))
            elif awaiting:
                status = "Awaiting approval"
            else:
                status = "Planned"
            stages.append(
                BotStage(
                    name=_stage_name(mode),
                    status=status,
                    detail=self._detail(jobs_by_mode.get(mode), mode),
                )
            )
        return tuple(stages)

    def _detail(self, job: Mapping[str, Any] | None, mode: str) -> str:
        if not job:
            return ""
        if mode == "synthetic":
            config = job.get("synthetic", {})
            if isinstance(config, Mapping):
                return " / ".join(
                    str(value)
                    for value in (config.get("adapter"), config.get("profile"))
                    if value
                )
        if mode == "hf_checkpoint":
            config = job.get("hf_checkpoint", {})
            if isinstance(config, Mapping):
                repo = str(config.get("repo", ""))
                revision = str(config.get("revision", ""))
                return f"{repo}@{revision[:12]}" if revision else repo
        return str(job.get("id", ""))

    def _metrics(self, results: Sequence[Mapping[str, Any]]) -> tuple[BotMetric, ...]:
        metrics: list[BotMetric] = []
        for result in sorted(results, key=lambda item: str(item.get("mode", ""))):
            mode = str(result.get("mode", "default"))
            values = result.get("metrics", {})
            if not isinstance(values, Mapping):
                continue
            for name, measurement in sorted(values.items()):
                if not isinstance(measurement, Mapping):
                    continue
                metrics.append(
                    BotMetric(
                        check=_stage_name(mode),
                        name=str(name),
                        base=measurement.get("base"),
                        head=measurement.get("head"),
                        change_pct=measurement.get("change_pct"),
                        verdict=str(measurement.get("verdict", "")),
                        unit=str(measurement.get("unit", "")),
                    )
                )
        return tuple(metrics)

    def _messages(
        self,
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        messages = [str(error.get("code", "unknown_error")) for error in errors]
        messages.extend(
            self._no_runner_message(result)
            for result in results
            if result.get("outcome") == "no_eligible_runner"
        )
        messages.extend(
            message
            for result in results
            if (message := self._findings_message(result)) is not None
        )
        if any(gate.get("status") == "awaiting_maintainer_approval" for gate in gates):
            messages.append(
                "Synthetic and checkpoint configuration passed static validation."
            )
            messages.append(
                "No Apple Silicon job starts until the protected environment is approved."
            )
        return tuple(messages)

    def _findings_message(self, result: Mapping[str, Any]) -> str | None:
        findings = result.get("findings")
        if not isinstance(findings, Mapping):
            return None
        cache = result.get("cache", {})
        cache_state = "cache reused" if cache.get("reused") else "downloaded"
        values = (
            ("prefill", findings.get("prefill_tps"), "tok/s"),
            ("decode", findings.get("decode_tps"), "tok/s"),
            ("TTFT", findings.get("ttft_ms"), "ms"),
            ("peak memory", findings.get("peak_memory_gib"), "GiB"),
        )
        measurements = "; ".join(
            f"{name} {value} {unit}"
            for name, value, unit in values
            if value is not None
        )
        output_hash = findings.get("output_hash")
        suffix = f"; output {output_hash}" if output_hash else ""
        return f"HF checkpoint findings ({cache_state}): {measurements}{suffix}."

    def _no_runner_message(self, result: Mapping[str, Any]) -> str:
        required = result.get("required_memory_gib", "unknown")
        required_disk = result.get("required_disk_gib")
        records = [
            item
            for key in ("attempts", "unavailable")
            for item in result.get(key, [])
            if isinstance(item, Mapping)
        ]
        summaries = [
            f"{item.get('device', 'unknown')} ({item.get('memory_gib', '?')} GiB): "
            f"{item.get('reason', 'declined')}"
            for item in records[:8]
        ]
        detail = "; ".join(summaries) if summaries else "no configured candidate"
        requirement = f"{required} GiB memory"
        if required_disk is not None:
            requirement += f" and {required_disk} GiB disk"
        return (
            f"No eligible Apple Silicon runner is available. Required: {requirement}. "
            f"Candidates: {detail}. Retry with /ci run."
        )


class BotOutput:
    """Compose one GitHub comment from independent CI component sections."""

    def __init__(
        self,
        record: Mapping[str, Any],
        components: Sequence[ComponentOutput] | None = None,
    ):
        self.record = record
        self.components = tuple(components or (ModelPathOutput(),))

    def render(self) -> str:
        sections = tuple(
            section
            for component in self.components
            for section in component.sections(self.record)
        )
        self._reject_unknown_components()
        lines = [
            "<!-- mlx-vlm-ci:status -->",
            "## MLX-VLM CI",
            "",
            f"Commit: `{_cell(self.record['head_sha'])}`  ",
            f"Status: **{_cell(self._status(sections))}**",
        ]
        run_url = self.record.get("run_url")
        if run_url:
            lines[-1] += f" · [workflow run]({_url(run_url)})"
        if sections:
            for section in sections:
                lines.extend(["", *section.render()])
            if self.record.get("kind") == "approved_job_plan":
                lines.extend(
                    ["", "The immutable job manifest is ready for runner dispatch."]
                )
        else:
            lines.extend(self._empty_output())
        return "\n".join(lines) + "\n"

    def _status(self, sections: Sequence[BotSection]) -> str:
        if not sections:
            return _label(str(self.record.get("outcome", "unknown")))
        statuses = {section.status for section in sections}
        for status in (
            "Blocked",
            "Test failed",
            "Regressed",
            "No eligible runner",
            "Infrastructure failed",
            "Cancelled",
            "Running",
            "Queued",
            "Awaiting maintainer approval",
            "Awaiting /ci run",
            "Ready for runner dispatch",
            "Passed",
        ):
            if status in statuses:
                return status
        return _label(str(self.record.get("outcome", "unknown")))

    def _empty_output(self) -> list[str]:
        errors = _items(self.record, "errors")
        if errors:
            lines = ["", "| Component | Subject | Problem |", "|---|---|---|"]
            lines.extend(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        error.get("component", "planner"),
                        error.get("subject", "pull_request"),
                        error.get("code", "unknown_error"),
                    )
                )
                + " |"
                for error in errors
            )
            lines.extend(
                ["", "Correct the configuration blockers and update the pull request."]
            )
            return lines
        return ["", "No Apple Silicon jobs are required for this change."]

    def _reject_unknown_components(self) -> None:
        supported = set().union(
            *(component.component_names for component in self.components)
        )
        encountered = {
            str(item.get("component"))
            for key in ("jobs", "gates", "results")
            for item in _items(self.record, key)
            if item.get("component")
        }
        encountered.update(
            str(error.get("component"))
            for error in _items(self.record, "errors")
            if error.get("component") not in {None, "planner"}
        )
        components = self.record.get("components", [])
        if isinstance(components, Sequence) and not isinstance(
            components, (str, bytes)
        ):
            encountered.update(str(component) for component in components if component)
        unknown = sorted(encountered - supported)
        if unknown:
            raise BotOutputError(
                "no bot output renderer for components: " + ", ".join(unknown)
            )


def _items(record: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = record.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _stage_name(mode: str) -> str:
    return {
        "synthetic": "Synthetic",
        "hf_checkpoint": "HF checkpoint",
    }.get(mode, mode.replace("_", " ").title())


def _label(value: str) -> str:
    return {
        "blocked": "Blocked",
        "awaiting_approval": "Awaiting maintainer approval",
        "ready": "Ready",
        "queued": "Queued",
        "running": "Running",
        "passed": "Passed",
        "regressed": "Regressed",
        "no_eligible_runner": "No eligible runner",
        "test_failure": "Test failed",
        "infrastructure_failure": "Infrastructure failed",
        "cancelled": "Cancelled",
    }.get(value, value.replace("_", " ").title())


def _measurement(value: Any, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        rendered = str(value)
    return f"{rendered} {unit}".rstrip()


def _change(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:+.2f}%"
    return str(value)


def _cell(value: Any) -> str:
    text = str(value).replace("@", "@\u200b")
    return (
        text.replace("|", "\\|")
        .replace("`", "\\`")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r", " ")
        .replace("\n", " ")[:200]
    )


def _url(value: Any) -> str:
    text = str(value)
    return text if text.startswith(("https://", "http://")) else "#"
