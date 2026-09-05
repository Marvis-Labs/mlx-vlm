from __future__ import annotations

import sys
from typing import Any

from ci.components.base import (
    ComponentContext,
    ComponentRegistration,
    ExecutionContext,
    PhaseRegistration,
)


def _planners(context: ComponentContext) -> tuple[Any, ...]:
    from ci.components.model_path import model_path_service
    from ci.mlp_change import GitSource, MLPChange

    return (
        MLPChange(
            context.config("components/mlp.yaml"),
            model_path_service(context),
            GitSource(context.repository),
        ),
    )


def _output():
    from ci.bot import MLPChangeOutput

    return MLPChangeOutput()


def _contract(context: ExecutionContext) -> list[str]:
    directory = context.config_directory
    return [
        sys.executable,
        str(directory / "mlp_contract_compare.py"),
        "--job",
        str(context.job_path),
        "--base",
        str(context.base),
        "--head",
        str(context.head),
        "--probe",
        str(directory / "mlp_contract_probe.py"),
    ]


REGISTRATION = ComponentRegistration(
    name="mlp_change",
    components=frozenset({"mlp_change"}),
    planner_factory=_planners,
    output_factory=_output,
    work=frozenset(),
    phases=(PhaseRegistration("mlp_contract", _contract),),
)
