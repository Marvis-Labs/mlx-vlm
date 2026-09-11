from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mlx_ci.repository.change_rules import ChangeContext, ChangeMatch
from mlx_ci.repository.components import ComponentContext, ComponentRegistration


class DocsChange:
    """Plan trusted GitHub-hosted documentation validation."""

    name = "docs_change"

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        paths = sorted({match.path for match in matches})
        return {
            "component": self.name,
            "checks": [
                {
                    "id": "docs",
                    "work_type": "Docs",
                    "component": self.name,
                    "execution_target": "github_hosted",
                    "handler": "docs",
                    "changed_paths": paths,
                }
            ],
            "jobs": [],
            "gates": [],
            "blocked": [],
        }


def _planners(context: ComponentContext) -> tuple[DocsChange, ...]:
    return (DocsChange(),)


REGISTRATION = ComponentRegistration(
    name="docs_change",
    components=frozenset({"docs_change"}),
    planner_factory=_planners,
    work=frozenset(),
)
