from __future__ import annotations

from typing import Any

from ci.components.base import ComponentContext, ComponentRegistration


def _planners(context: ComponentContext) -> tuple[Any, ...]:
    from ci.mlp_change import GitSource
    from ci.security_change import SecurityChange

    return (
        SecurityChange(
            context.config("components/security.yaml"),
            GitSource(context.repository),
        ),
    )


def _output():
    from ci.bot import SecurityChangeOutput

    return SecurityChangeOutput()


REGISTRATION = ComponentRegistration(
    name="security_change",
    components=frozenset({"security_change"}),
    planner_factory=_planners,
    output_factory=_output,
    work=frozenset(),
)
