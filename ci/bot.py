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
    stage_order = ("mlp_contract", "synthetic", "hf_checkpoint")

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
        messages = self._messages(jobs, gates, errors, results)
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
            pending = gate.get("pending_work")
            if isinstance(pending, Mapping):
                jobs.append(pending)
        unique: dict[str, Mapping[str, Any]] = {}
        for job in jobs:
            identifier = str(job.get("id", model))
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
            "coalesced",
            "improved",
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
        jobs_by_mode = {phase: job for job in jobs for phase in self._job_phases(job)}
        results_by_mode = {
            str(result["phase"]): result
            for result in self._phase_results(results)
            if result.get("phase")
        }
        for result in results:
            if result.get("phases") or result.get("mode"):
                continue
            for phase in jobs_by_mode:
                results_by_mode.setdefault(phase, result)
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

    def _job_phases(self, job: Mapping[str, Any]) -> tuple[str, ...]:
        phases = job.get("phases")
        if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
            return tuple(str(phase) for phase in phases)
        mode = job.get("mode")
        return (str(mode),) if mode else ()

    def _phase_results(
        self, results: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        expanded: list[dict[str, Any]] = []
        for result in results:
            phases = result.get("phases")
            if isinstance(phases, Mapping):
                for name, phase_result in phases.items():
                    if not isinstance(phase_result, Mapping):
                        continue
                    item = dict(phase_result)
                    item["phase"] = str(name)
                    if name == "hf_checkpoint" and "cache" not in item:
                        item["cache"] = result.get("cache", {})
                    expanded.append(item)
                continue
            mode = result.get("mode")
            if mode:
                item = dict(result)
                item["phase"] = str(mode)
                expanded.append(item)
        return tuple(expanded)

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
        if mode == "mlp_contract":
            config = job.get("mlp_contract", {})
            if isinstance(config, Mapping):
                return ", ".join(str(value) for value in config.get("symbols", []))
        if mode == "hf_checkpoint":
            config = job.get("hf_checkpoint", {})
            if isinstance(config, Mapping):
                repo = str(config.get("repo", ""))
                revision = str(config.get("revision", ""))
                return f"{repo}@{revision[:12]}" if revision else repo
        return str(job.get("id", ""))

    def _metrics(self, results: Sequence[Mapping[str, Any]]) -> tuple[BotMetric, ...]:
        metrics: list[BotMetric] = []
        for result in sorted(
            self._phase_results(results), key=lambda item: str(item.get("phase", ""))
        ):
            mode = str(result.get("phase", "default"))
            values = result.get("metrics", {})
            findings = result.get("findings", {})
            if not isinstance(values, Mapping) or not values:
                values = (
                    findings.get("metrics", {}) if isinstance(findings, Mapping) else {}
                )
            if not isinstance(values, Mapping):
                continue
            correctness = (
                findings.get("correctness", {}) if isinstance(findings, Mapping) else {}
            )
            advisory = (
                isinstance(correctness, Mapping) and correctness.get("match") is False
            )
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
                        verdict=(
                            "advisory"
                            if advisory
                            else str(measurement.get("verdict", ""))
                        ),
                        unit=str(measurement.get("unit", "")),
                    )
                )
        return tuple(metrics)

    def _messages(
        self,
        jobs: Sequence[Mapping[str, Any]],
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        messages = [str(error.get("code", "unknown_error")) for error in errors]
        messages.extend(
            f"{_stage_name(str(phase))} unavailable: {reason}."
            for job in jobs
            for phase, reason in (
                job.get("unavailable_phases", {}).items()
                if isinstance(job.get("unavailable_phases"), Mapping)
                else ()
            )
        )
        messages.extend(
            message
            for result in results
            for message in self._execution_metadata(result)
        )
        messages.extend(
            message
            for result in results
            if result.get("outcome") == "no_eligible_runner"
            for message in self._no_runner_messages(result)
        )
        messages.extend(
            message
            for result in self._phase_results(results)
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

    def _execution_metadata(self, result: Mapping[str, Any]) -> tuple[str, ...]:
        selected = result.get("selected_device")
        runner = (
            selected.get("name", "unknown")
            if isinstance(selected, Mapping)
            else result.get("device", "not allocated")
        )
        if result.get("outcome") == "coalesced":
            runner = (
                "not allocated; coalesced with attempt "
                f"{result.get('owner_attempt_id', 'unknown')}"
            )
        cache = result.get("cache", {})
        if isinstance(cache, Mapping) and cache:
            cache_state = (
                "reused"
                if cache.get("reused")
                else str(cache.get("after", cache.get("before", "not checked")))
            )
        else:
            cache_state = "not checked"
        correctness_values: list[bool] = []
        has_metrics = False
        for phase in self._phase_results((result,)):
            findings = phase.get("findings", {})
            if not isinstance(findings, Mapping):
                continue
            correctness = findings.get("correctness", {})
            if isinstance(correctness, Mapping) and isinstance(
                correctness.get("match"), bool
            ):
                correctness_values.append(correctness["match"])
            has_metrics = has_metrics or isinstance(findings.get("metrics"), Mapping)
        correctness_state = (
            "failed"
            if False in correctness_values
            else "passed" if correctness_values else "not reported"
        )
        performance_state = (
            "advisory"
            if False in correctness_values and has_metrics
            else "reported" if has_metrics else "not run"
        )
        return (
            f"Runner: {runner}.",
            f"Cache: {cache_state}.",
            f"Correctness: {correctness_state}.",
            f"Performance: {performance_state}.",
            f"Terminal state: {_label(str(result.get('outcome', 'unknown')))}.",
        )

    def _findings_message(self, result: Mapping[str, Any]) -> str | None:
        findings = result.get("findings")
        if not isinstance(findings, Mapping):
            return None
        phase = str(result.get("phase", "hf_checkpoint"))
        error = findings.get("error")
        if error:
            return f"{_stage_name(phase)} comparison failed: {error}."
        correctness = findings.get("correctness", {})
        if isinstance(correctness, Mapping) and correctness.get("match") is False:
            if phase == "mlp_contract":
                return "MLP output, parameter structure, or reference formula did not match main."
            if phase == "synthetic":
                return "Synthetic output or parameter structure did not match main."
            cache = result.get("cache", {})
            cache_state = "cache reused" if cache.get("reused") else "downloaded"
            return (
                "Checkpoint output mismatch: main "
                f"{correctness.get('base_output_hash')} versus PR "
                f"{correctness.get('head_output_hash')}. Performance measurements "
                f"are advisory because correctness failed ({cache_state})."
            )
        if phase in {"mlp_contract", "synthetic"}:
            if isinstance(correctness, Mapping) and correctness.get("match") is True:
                return (
                    "MLP structure, output, and reference formula match main."
                    if phase == "mlp_contract"
                    else "Synthetic structure and output match main."
                )
            return None
        cache = result.get("cache", {})
        cache_state = "cache reused" if cache.get("reused") else "downloaded"
        head = findings.get("head", findings)
        if not isinstance(head, Mapping):
            return f"HF checkpoint comparison completed ({cache_state})."
        values = (
            ("prefill", head.get("prefill_tps"), "tok/s"),
            ("decode", head.get("decode_tps"), "tok/s"),
            ("TTFT", head.get("ttft_ms"), "ms"),
            ("peak memory", head.get("peak_memory_gib"), "GiB"),
        )
        measurements = "; ".join(
            f"{name} {value} {unit}"
            for name, value, unit in values
            if value is not None
        )
        output_hash = head.get("output_hash")
        suffix = f"; output {output_hash}" if output_hash else ""
        return f"HF checkpoint findings ({cache_state}): {measurements}{suffix}."

    def _no_runner_messages(self, result: Mapping[str, Any]) -> tuple[str, ...]:
        required = result.get("required_memory_gib", "unknown")
        required_disk = result.get("required_disk_gib")
        records = [
            item
            for key in ("attempts", "unavailable")
            for item in result.get(key, [])
            if isinstance(item, Mapping)
        ]
        summaries = [self._runner_unavailable_summary(item) for item in records[:8]]
        requirement = f"{required} GiB memory"
        if required_disk is not None:
            requirement += f" and {required_disk} GiB disk"
        candidates = summaries or ["no configured candidate"]
        return (
            f"No eligible Apple Silicon runner is available. Required: {requirement}.",
            *(f"Candidate: {summary}." for summary in candidates),
            "Retry with /ci run.",
        )

    def _runner_unavailable_summary(self, item: Mapping[str, Any]) -> str:
        prefix = (
            f"{item.get('device', 'unknown')} " f"({item.get('memory_gib', '?')} GiB)"
        )
        if item.get("reason") != "leased":
            return f"{prefix}: {item.get('reason', 'declined')}"
        owner = item.get("attempt_id", "unknown")
        expiry = item.get("expires_at", "unknown")
        return f"{prefix}: leased by attempt {owner} until {expiry}"


class MLPChangeOutput:
    """Summarize the MLP symbols that expanded into ModelPath work."""

    component_names = frozenset({"mlp_change"})

    def sections(self, record: Mapping[str, Any]) -> Sequence[BotSection]:
        jobs = [
            job
            for job in _items(record, "jobs")
            if job.get("component") == "model_path"
            and isinstance(job.get("origins"), Sequence)
        ]
        results = {
            str(result.get("job_id")): result for result in _items(record, "results")
        }
        by_symbol: dict[str, list[Mapping[str, Any]]] = {}
        for job in jobs:
            for origin in job.get("origins", []):
                if (
                    not isinstance(origin, Mapping)
                    or origin.get("change_type") != "MLPChange"
                ):
                    continue
                by_symbol.setdefault(str(origin.get("symbol")), []).append(job)
        return tuple(
            self._section(symbol, symbol_jobs, results)
            for symbol, symbol_jobs in sorted(by_symbol.items())
        )

    def _section(
        self,
        symbol: str,
        jobs: Sequence[Mapping[str, Any]],
        results: Mapping[str, Mapping[str, Any]],
    ) -> BotSection:
        outcomes = {
            str(results[str(job.get("id"))].get("outcome"))
            for job in jobs
            if str(job.get("id")) in results
        }
        status = "Awaiting /ci run"
        for outcome in (
            "test_failure",
            "regressed",
            "no_eligible_runner",
            "infrastructure_failure",
            "cancelled",
            "running",
            "queued",
            "improved",
            "passed",
        ):
            if outcome in outcomes:
                status = _label(outcome)
                break
        models = sorted(str(job.get("model")) for job in jobs)
        checkpoint_count = sum("hf_checkpoint" in job.get("phases", []) for job in jobs)
        return BotSection(
            title=symbol,
            component="MLPChange",
            status=status,
            paths=("mlx_vlm/models/mlp.py",),
            stages=(
                BotStage(
                    "Selected consumers",
                    "Planned",
                    f"{len(models)} ModelPath jobs; {checkpoint_count} with HF checkpoints",
                ),
            ),
            metrics=(),
            messages=("Models: " + ", ".join(models),),
        )


class KVCacheChangeOutput:
    """Render one independently scheduled section per KV-cache storage profile."""

    component_names = frozenset({"kv_cache_change"})

    def sections(self, record: Mapping[str, Any]) -> Sequence[BotSection]:
        jobs = [
            job
            for job in _items(record, "jobs")
            if job.get("component") == "kv_cache_change"
        ]
        results = {
            str(result.get("job_id")): result
            for result in _items(record, "results")
            if result.get("component") == "kv_cache_change"
        }
        return tuple(
            self._section(record, job, results.get(str(job.get("id")))) for job in jobs
        )

    def _section(
        self,
        record: Mapping[str, Any],
        job: Mapping[str, Any],
        result: Mapping[str, Any] | None,
    ) -> BotSection:
        profile = str(job.get("profile", "unknown"))
        contract = job.get("kv_cache_contract", {})
        implementations = (
            contract.get("implementations", []) if isinstance(contract, Mapping) else []
        )
        phase = self._phase(result)
        status = self._status(record, result, bool(job))
        phase_status = (
            _label(str(phase.get("outcome", "unknown")))
            if phase is not None
            else (
                _label(str(result.get("outcome", "unknown")))
                if result is not None
                else "Planned"
            )
        )
        detail = ", ".join(str(value) for value in implementations)
        return BotSection(
            title=profile,
            component="KVCacheChange",
            status=status,
            paths=tuple(sorted(str(path) for path in job.get("changed_paths", []))),
            stages=(BotStage("KV cache contract", phase_status, detail),),
            metrics=(),
            messages=self._messages(job, result, phase),
        )

    @staticmethod
    def _phase(result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if result is None:
            return None
        phases = result.get("phases")
        if not isinstance(phases, Mapping):
            return None
        phase = phases.get("kv_cache_contract")
        return phase if isinstance(phase, Mapping) else None

    @staticmethod
    def _status(
        record: Mapping[str, Any], result: Mapping[str, Any] | None, has_job: bool
    ) -> str:
        if result is not None:
            return _label(str(result.get("outcome", "unknown")))
        if record.get("kind") == "approved_job_plan" and has_job:
            return "Ready for runner dispatch"
        return "Awaiting /ci run" if has_job else "No jobs"

    def _messages(
        self,
        job: Mapping[str, Any],
        result: Mapping[str, Any] | None,
        phase: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        messages = [
            f"Head: {job.get('head_sha', 'unknown')}; trusted contract: {job.get('contract_sha', 'unknown')}."
        ]
        if result is None:
            return tuple(messages)
        if result.get("outcome") == "no_eligible_runner":
            messages.append(
                f"No eligible runner; requires {result.get('required_memory_gib', 'unknown')} GiB memory."
            )
            return tuple(messages)
        selected = result.get("selected_device")
        runner = (
            selected.get("name", "unknown")
            if isinstance(selected, Mapping)
            else result.get("device", "not reported")
        )
        messages.append(f"Runner: {runner}.")
        findings = (
            phase.get("findings") if phase is not None else result.get("findings")
        )
        if isinstance(findings, Mapping):
            messages.extend(self._run_messages(findings))
            error = findings.get("error")
            if error:
                messages.append(f"Contract execution failed: {error}.")
        messages.append(
            f"Terminal state: {_label(str(result.get('outcome', 'unknown')))}."
        )
        return tuple(messages)

    @staticmethod
    def _run_messages(findings: Mapping[str, Any]) -> tuple[str, ...]:
        messages: list[str] = []
        for case in findings.get("cases", []):
            if not isinstance(case, Mapping):
                continue
            runs = [
                str(run.get("sequence"))
                for run in case.get("runs", [])
                if isinstance(run, Mapping) and run.get("sequence")
            ]
            deterministic = [name for name in runs if "-state-machine-" not in name]
            state_machines = len(runs) - len(deterministic)
            shown = deterministic[:4]
            run_summary = ", ".join(shown)
            hidden = len(deterministic) - len(shown)
            if hidden:
                run_summary += f", and {hidden} more deterministic runs"
            if state_machines:
                suffix = f"{state_machines} seeded state-machine runs"
                run_summary = f"{run_summary}, {suffix}" if run_summary else suffix
            messages.append(
                f"{case.get('case', 'cache')}: {case.get('checks', 0)} checks; "
                + (run_summary if runs else "no runs reported")
                + "."
            )
            failures = case.get("failures", [])
            if isinstance(failures, Sequence) and failures:
                first = failures[0]
                if isinstance(first, Mapping):
                    messages.append(
                        "First failure: "
                        f"{first.get('sequence', 'unknown')} step {first.get('step', '?')} "
                        f"{first.get('characteristic', first.get('reason', 'unknown'))}."
                    )
        return tuple(messages)


class BotOutput:
    """Compose one GitHub comment from independent CI component sections."""

    def __init__(
        self,
        record: Mapping[str, Any],
        components: Sequence[ComponentOutput] | None = None,
    ):
        self.record = record
        if components is None:
            from ci.components.registry import outputs

            components = outputs()
        self.components = tuple(components)

    def render(self) -> str:
        sections = tuple(
            section
            for component in self.components
            for section in component.sections(self.record)
        )
        self._reject_unknown_components()
        lines = [
            self._marker(),
            f"Commit: `{_cell(self.record['head_sha'])}`  ",
            f"Status: **{_cell(self._status(sections))}**",
        ]
        if self.record.get("kind") == "ci_execution" and self.record.get("attempt_id"):
            lines.insert(2, f"Attempt: `{_cell(self.record['attempt_id'])}`  ")
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

    def _marker(self) -> str:
        attempt_id = self.record.get("attempt_id")
        if self.record.get("kind") == "ci_execution" and attempt_id:
            return f"<!-- mlx-vlm:ci:attempt:{_cell(attempt_id)} -->"
        return "<!-- mlx-vlm:ci:plan -->"

    def _status(self, sections: Sequence[BotSection]) -> str:
        if self.record.get("outcome") == "blocked":
            return "Blocked"
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
            "Coalesced",
            "Awaiting maintainer approval",
            "Awaiting /ci run",
            "Ready for runner dispatch",
            "Improved",
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
        "kv_cache_contract": "KV cache contract",
        "mlp_contract": "MLP contract",
        "synthetic": "Synthetic",
        "hf_checkpoint": "HF checkpoint",
    }.get(mode, mode.replace("_", " ").title())


def _label(value: str) -> str:
    return {
        "blocked": "Blocked",
        "awaiting_approval": "Awaiting maintainer approval",
        "ready": "Ready",
        "queued": "Queued",
        "coalesced": "Coalesced",
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
