from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlx_ci.repository.components import ComponentContext, ExecutionContext

from ci.docs import REGISTRATION as DOCS
from ci.model_path.component import REGISTRATION as MODEL_PATH

REGISTRATIONS = (DOCS, MODEL_PATH)

DISPLAY_LABELS = {
    "decode_tps": "Decode throughput",
    "embedding_latency_ms": "Embedding latency",
    "embedding_tps": "Embedding throughput",
    "hf_checkpoint": "HF checkpoint",
    "peak_memory_gib": "Peak memory",
    "prefill_tps": "Prefill throughput",
    "synthetic": "Synthetic",
    "ttft_ms": "TTFT",
    "wall_ms": "Wall time",
}

FAILURE_MESSAGES = {
    "checkpoint_not_found": "The configured checkpoint or revision was not found.",
    "access_denied": (
        "The configured checkpoint requires access the CI runner does not have."
    ),
    "disk_full": "The selected runner does not have enough disk space.",
    "network_transient": (
        "The checkpoint download failed temporarily; retry with /ci run."
    ),
    "checkpoint_policy_failed": (
        "The runner rejected the checkpoint because it failed integrity policy."
    ),
    "checkpoint_internal_error": (
        "The runner could not safely prepare the checkpoint."
    ),
}


def planners(
    config_directory: Path,
    repository: Path,
    contributor_config_directory: Path | None = None,
) -> tuple[Any, ...]:
    context = ComponentContext(
        config_directory,
        repository,
        contributor_config_directory,
    )
    return tuple(
        planner
        for registration in REGISTRATIONS
        for planner in registration.planner_factory(context)
    )


def display_labels() -> Mapping[str, str]:
    return DISPLAY_LABELS


def failure_messages() -> Mapping[str, str]:
    return FAILURE_MESSAGES


def phase_environment() -> Mapping[str, str]:
    return {"CI_NETWORK_DISABLED": "1"}


def validate_job(job: Mapping[str, Any]) -> None:
    common = {
        "id",
        "work_type",
        "component",
        "subject",
        "model",
        "changed_paths",
        "phases",
        "required_memory_gib",
        "required_disk_gib",
        "repository",
        "base_sha",
        "head_sha",
        "contract_sha",
        "manifest_digest",
    }
    allowed = common | set().union(
        *(registration.job_fields for registration in REGISTRATIONS)
    )
    if unexpected := sorted(set(job) - allowed):
        raise ValueError(
            "work manifest contains unregistered fields: " + ", ".join(unexpected)
        )
    work = (job.get("work_type"), job.get("component"))
    supported_work = {
        item for registration in REGISTRATIONS for item in registration.work
    }
    if work not in supported_work:
        raise ValueError(f"unregistered work item: {work!r}")
    phases = job.get("phases", [])
    if phases != ["synthetic", "hf_checkpoint"]:
        raise ValueError("model path phases must be synthetic then hf_checkpoint")
    subject = job.get("subject")
    if job.get("model") != subject or job.get("id") != f"model_path:{subject}":
        raise ValueError("model path identity is inconsistent")
    paths = job.get("changed_paths")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path for path in paths)
    ):
        raise ValueError("model path changed_paths are invalid")
    synthetic = job.get("synthetic")
    if not isinstance(synthetic, Mapping) or set(synthetic) != {
        "adapter",
        "profile",
    }:
        raise ValueError("model path synthetic configuration is invalid")
    if any(not isinstance(value, str) or not value for value in synthetic.values()):
        raise ValueError("model path synthetic configuration is invalid")
    from mlx_ci.repository.checkpoint_policy import validate_checkpoint

    validate_checkpoint(job.get("hf_checkpoint", {}))
    scenarios = job.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or any(not isinstance(scenario, str) or not scenario for scenario in scenarios)
    ):
        raise ValueError("model path scenarios are invalid")


def phase_commands(context: ExecutionContext) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for registration in REGISTRATIONS:
        for phase in registration.phases:
            if phase.name in commands:
                raise ValueError(f"duplicate phase registration: {phase.name}")
            commands[phase.name] = phase.command(context)
    return commands


def contributor_config_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for registration in REGISTRATIONS
                for path in registration.contributor_configs
            }
        )
    )


def validate_gate(gate: Mapping[str, Any]) -> None:
    component = str(gate.get("component", ""))
    registrations = [
        registration
        for registration in REGISTRATIONS
        if component in registration.components
        and registration.gate_validator is not None
    ]
    if len(registrations) != 1:
        raise ValueError(f"no unique gate validator for component: {component}")
    validator = registrations[0].gate_validator
    if validator is None:
        raise ValueError(f"component does not support approval gates: {component}")
    validator(gate)
