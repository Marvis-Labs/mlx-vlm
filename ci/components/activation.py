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
    from ci.activation_change import ActivationChange
    from ci.mlp_change import GitSource

    return (
        ActivationChange(
            context.config("components/activation.yaml"),
            GitSource(context.repository),
        ),
    )


def _output():
    from ci.bot import ActivationChangeOutput

    return ActivationChangeOutput()


def _contract(context: ExecutionContext) -> list[str]:
    directory = context.config_directory
    return [
        sys.executable,
        str(directory / "activation_contract_compare.py"),
        "--job",
        str(context.job_path),
        "--base",
        str(context.base),
        "--head",
        str(context.head),
        "--probe",
        str(directory / "activation_contract_probe.py"),
    ]


REGISTRATION = ComponentRegistration(
    name="activation_change",
    components=frozenset({"activation_change"}),
    planner_factory=_planners,
    output_factory=_output,
    work=frozenset({("ActivationChange", "activation_change")}),
    phases=(PhaseRegistration("activation_contract", _contract),),
    job_fields=frozenset({"activation_contract"}),
)
