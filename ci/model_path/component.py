from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

from mlx_ci.repository.components import (
    ComponentContext,
    ComponentRegistration,
    ExecutionContext,
    PhaseRegistration,
)

SYNTHETIC_ADAPTERS = frozenset(
    {
        "bert",
        "deepseek_vl_v2",
        "granite_vision",
        "internvl_chat",
        "qwen2_5_vl",
        "qwen2_vl",
    }
)


def resource_requirements(
    checkpoint: Mapping[str, Any] | None,
) -> tuple[int, int]:
    if checkpoint is None:
        return 8, 2
    weight = checkpoint.get("weight")
    if not isinstance(weight, Mapping):
        raise ValueError("checkpoint has no weight metadata")  # noqa: TRY004
    weight_bytes = weight.get("bytes")
    if not isinstance(weight_bytes, int) or weight_bytes <= 0:
        raise ValueError("checkpoint weight bytes must be positive")
    weights_gib = weight_bytes / 2**30
    return (
        max(8, math.ceil(weights_gib * 1.5 + 4)),
        max(4, math.ceil(weights_gib * 1.25 + 2)),
    )


def build_model_path(context: ComponentContext):
    from ci.model_path.planning import ModelPath

    return ModelPath(
        context.config("config/models.yaml", contributor=True),
        context.config("config/scenarios.yaml", contributor=True),
        supported_synthetic_adapters=SYNTHETIC_ADAPTERS,
    )


def _planners(context: ComponentContext) -> tuple[Any, ...]:
    from ci.model_path.planning import NewModelPath

    model_path = build_model_path(context)
    return NewModelPath(model_path), model_path


def _synthetic(context: ExecutionContext) -> list[str]:
    directory = context.config_directory
    return [
        sys.executable,
        str(directory / "model_path/synthetic_compare.py"),
        "--job",
        str(context.job_path),
        "--profiles",
        str(directory / "config/models.yaml"),
        "--base",
        str(context.base),
        "--head",
        str(context.head),
        "--probe",
        str(directory / "model_path/synthetic_probe.py"),
    ]


def _hf_checkpoint(context: ExecutionContext) -> list[str]:
    directory = context.config_directory
    return [
        sys.executable,
        str(directory / "model_path/checkpoint_compare.py"),
        "--job",
        str(context.job_path),
        "--scenarios",
        str(directory / "config/scenarios.yaml"),
        "--base",
        str(context.base),
        "--head",
        str(context.head),
        "--probe",
        str(directory / "model_path/checkpoint_probe.py"),
        "--image",
        str(directory / "assets/cat.jpg"),
        "--max-tokens",
        "16",
    ]


def _validate_gate(gate: Mapping[str, Any]) -> None:
    pending_work = gate.get("pending_work")
    requested = gate.get("requested_phases")
    if not isinstance(pending_work, Mapping):
        raise ValueError("approval gate has no pending work")  # noqa: TRY004
    if not isinstance(requested, list) or not requested:
        raise ValueError("approval gate has no requested phases")
    if (
        pending_work.get("work_type") != "ModelPath"
        or pending_work.get("component") != "model_path"
        or pending_work.get("model") != gate.get("model")
        or pending_work.get("phases") != requested
    ):
        raise ValueError("approval gate pending work exceeds its scope")


REGISTRATION = ComponentRegistration(
    name="model_path",
    components=frozenset({"model_path", "new_model_path"}),
    planner_factory=_planners,
    work=frozenset({("ModelPath", "model_path")}),
    phases=(
        PhaseRegistration("synthetic", _synthetic),
        PhaseRegistration("hf_checkpoint", _hf_checkpoint),
    ),
    contributor_configs=frozenset({"config/models.yaml", "config/scenarios.yaml"}),
    gate_validator=_validate_gate,
    job_fields=frozenset(
        {
            "synthetic",
            "hf_checkpoint",
            "scenarios",
            "unavailable_phases",
        }
    ),
)
